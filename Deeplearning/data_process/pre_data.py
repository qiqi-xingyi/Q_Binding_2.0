# --*-- conding:utf-8 --*--
# @time:7/29/25 11:06
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# pre_data.py  ──  QM9  .xyz  →  (h, g) + Pauli  HDF5

import os, glob, json, argparse, multiprocessing as mp
import numpy as np, h5py
from tqdm import tqdm
from pyscf import gto, scf, ao2mo
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems     import ElectronicStructureProblem
from qiskit_nature.second_q.mappers      import JordanWignerMapper


def read_xyz(path):
    """return Z(np.int16[N]), coords(np.float32[N,3])."""
    with open(path) as fh:
        lines = fh.readlines()
    n = int(lines[0]); Z, C = [], []
    for ln in lines[2:2+n]:
        sym, x, y, z = ln.split()[:4]
        Z.append(gto.mole._charge(sym))
        C.append([float(x), float(y), float(z)])
    return np.array(Z, np.int16), np.array(C, np.float32)

def indep_mask(n):
    """chemists’ notation, keep (p≥q, r≥s, pq≥rs)."""
    idx = np.triu_indices(n)
    mask = np.zeros((n, n, n, n), bool)
    for p, q in zip(*idx):
        for r, s in zip(*idx):
            mask[p, q, r, s] = (p*n + q) >= (r*n + s)
    return mask.reshape(-1)

mask_cache = {}  # n_orb -> bool mask

def calc_integrals(Z, C, basis):
    mol = gto.M(atom=list(zip(Z, C)),
                basis=basis, unit="Angstrom", cart=False)
    n = mol.nao_nr()
    mf = scf.RHF(mol).run()
    h_vec = mf.get_hcore()[np.triu_indices(n)].astype("float32")

    eri = ao2mo.restore(1, mol.intor("int2e"), n).reshape(-1)
    if n not in mask_cache:
        mask_cache[n] = indep_mask(n)
    g_vec = eri[mask_cache[n]].astype("float32")
    return h_vec, g_vec, n

def build_pauli(h_vec, g_vec, n, eps):
    """
    (h,g) -> SparsePauliOp via Jordan-Wigner mapping.
    returns: strings(bytes[K]), coeffs(float32[K])
    """
    # reconstruct full tensors
    h = np.zeros((n, n), np.float64)
    h[np.triu_indices(n)] = h_vec
    h = h + h.T - np.diag(np.diag(h))

    g = np.zeros((n, n, n, n), np.float64)
    g.reshape(-1)[mask_cache[n]] = g_vec
    # symmetrize (chemists)
    g = (g + g.transpose(1,0,2,3) + g.transpose(2,3,0,1)) / 3.0

    elH = ElectronicEnergy.from_raw_integrals(h, g,
                                              num_spatial_orbitals=n)
    sec_op = ElectronicStructureProblem(elH).second_q_ops()["ElectronicEnergy"]

    mapper   = JordanWignerMapper()
    pauli_op = mapper.map(sec_op)          # ← 新接口

    sel = np.abs(pauli_op.coeffs.real) > eps
    strings = np.array([p.to_label() for p in pauli_op.paulis[sel]],
                       dtype="S")          # bytes
    coeffs  = pauli_op.coeffs.real[sel].astype("float32")
    return strings, coeffs

# ─────────────────────────  Worker  ───────────────────────────
def worker(arg):
    xyz_path, out_int, out_pau, basis, eps = arg
    stem = os.path.splitext(os.path.basename(xyz_path))[0]   # gdb1
    Z, C = read_xyz(xyz_path)

    h, g, n = calc_integrals(Z, C, basis)
    p_strs, p_coeffs = build_pauli(h, g, n, eps)

    with h5py.File(f"{out_int}/{stem}.h5", "w") as f:
        f["Z"] = Z; f["pos"] = C
        f["h"] = h; f["g"]  = g
        f.attrs.update(n_orb=n, basis=basis)


    with h5py.File(f"{out_pau}/{stem}.h5", "w") as f:
        f["strings"] = p_strs
        f["coeffs"]  = p_coeffs
        f.attrs.update(mapper="JordanWigner", eps=eps,
                       n_orb=n, basis=basis)
    return stem

# ##  Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",   required=True, help="dir of *.xyz")
    ap.add_argument("--dst",   default="data", help="parent output dir")
    ap.add_argument("--basis", default="sto3g")
    ap.add_argument("--eps",   type=float, default=1e-4)
    ap.add_argument("--n_proc",type=int, default=8)
    args = ap.parse_args()

    out_int = f"{args.dst}/integrals_h5"
    out_pau = f"{args.dst}/paulis_h5"
    os.makedirs(out_int, exist_ok=True)
    os.makedirs(out_pau, exist_ok=True)

    xyz_paths = sorted(glob.glob(f"{args.src}/*.xyz"))
    pool = mp.Pool(args.n_proc)
    bar  = tqdm(total=len(xyz_paths), desc="XYZ → (h,g)+Pauli")
    for _ in pool.imap_unordered(lambda p: worker((p,out_int,out_pau,
                                                   args.basis,args.eps)),
                                 xyz_paths):
        bar.update()
    pool.close(); pool.join(); bar.close()

    files = sorted(glob.glob(f"{out_int}/*.h5"))
    np.random.shuffle(files)
    n = len(files)
    splits = {"train": files[:int(.8*n)],
              "val":   files[int(.8*n):int(.9*n)],
              "test":  files[int(.9*n):]}
    json.dump(splits, open(f"{args.dst}/splits.json", "w"))
    print("Finished:", n, "samples")

if __name__ == "__main__":
    main()

