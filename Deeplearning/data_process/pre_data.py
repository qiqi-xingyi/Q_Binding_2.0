# --*-- conding:utf-8 --*--
# @time:7/29/25 11:06
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# pre_data.py  ──  QM9  .xyz  →  (h, g) + Pauli  HDF5

import os, glob, json, multiprocessing as mp
import numpy as np, h5py
from tqdm import tqdm
from pyscf import gto, scf, ao2mo
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems     import ElectronicStructureProblem
from qiskit_nature.second_q.mappers      import JordanWignerMapper

# ─────────────── Hardcoded Settings ───────────────
SRC_DIR = "QM9/archive"     # directory containing .xyz files
DST_DIR = "QM9/data"        # parent output directory
BASIS   = "cc-pVTZ"           # basis set
EPS     = 1e-4                # Pauli coefficient threshold
N_PROC  = 8                   # number of parallel processes

# Derived paths
OUT_INT = os.path.join(DST_DIR, "integrals_h5")
OUT_PAU = os.path.join(DST_DIR, "paulis_h5")

# ─────────────────── Utility Functions ───────────────────

def read_xyz(path):
    """Return Z(int16[N]), coords(float32[N,3])."""
    with open(path) as fh:
        lines = fh.readlines()
    n = int(lines[0]); Z, C = [], []
    for ln in lines[2:2+n]:
        sym, x, y, z = ln.split()[:4]
        Z.append(gto.mole._charge(sym))
        C.append([float(x), float(y), float(z)])
    return np.array(Z, np.int16), np.array(C, np.float32)


def indep_mask(n):
    """Chemists' notation mask for two-electron integrals."""
    idx = np.triu_indices(n)
    mask = np.zeros((n, n, n, n), bool)
    for p, q in zip(*idx):
        for r, s in zip(*idx):
            mask[p, q, r, s] = (p*n + q) >= (r*n + s)
    return mask.reshape(-1)

mask_cache = {}

def calc_integrals(Z, C, basis):
    mol = gto.M(atom=list(zip(Z, C)), basis=basis, unit="Angstrom", cart=False)
    n   = mol.nao_nr()
    mf  = scf.RHF(mol).run()
    h_vec = mf.get_hcore()[np.triu_indices(n)].astype("float32")
    eri   = ao2mo.restore(1, mol.intor("int2e"), n).reshape(-1)
    if n not in mask_cache:
        mask_cache[n] = indep_mask(n)
    g_vec = eri[mask_cache[n]].astype("float32")
    return h_vec, g_vec, n


def build_pauli(h_vec, g_vec, n, eps):
    h = np.zeros((n, n), np.float64)
    h[np.triu_indices(n)] = h_vec
    h = h + h.T - np.diag(np.diag(h))

    g = np.zeros((n, n, n, n), np.float64)
    g.reshape(-1)[mask_cache[n]] = g_vec
    g = (g + g.transpose(1,0,2,3) + g.transpose(2,3,0,1)) / 3.0

    elH    = ElectronicEnergy.from_raw_integrals(h, g)
    sec_op = ElectronicStructureProblem(elH).second_q_ops()["ElectronicEnergy"]

    mapper   = JordanWignerMapper()
    pauli_op = mapper.map(sec_op)

    sel     = np.abs(pauli_op.coeffs.real) > eps
    strings = np.array([p.to_label() for p in pauli_op.paulis[sel]], dtype="S")
    coeffs  = pauli_op.coeffs.real[sel].astype("float32")
    return strings, coeffs

# ───────────────────────── Worker ─────────────────────────
def worker(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    Z, C = read_xyz(path)
    h, g, n = calc_integrals(Z, C, BASIS)
    p_strs, p_coeffs = build_pauli(h, g, n, EPS)

    # save integrals
    with h5py.File(os.path.join(OUT_INT, f"{stem}.h5"), "w") as f:
        f["Z"] = Z
        f["pos"] = C
        f["h"] = h
        f["g"] = g
        f.attrs.update(n_orb=n, basis=BASIS)

    # save pauli
    with h5py.File(os.path.join(OUT_PAU, f"{stem}.h5"), "w") as f:
        f["strings"] = p_strs
        f["coeffs"]  = p_coeffs
        f.attrs.update(mapper="JordanWigner", eps=EPS, n_orb=n, basis=BASIS)
    return stem

# ───────────────────────── Main ───────────────────────────
def main():
    # Ensure output dirs exist
    os.makedirs(OUT_INT, exist_ok=True)
    os.makedirs(OUT_PAU, exist_ok=True)

    xyz_paths = sorted(glob.glob(os.path.join(SRC_DIR, "*.xyz")))
    pool = mp.Pool(N_PROC)
    bar  = tqdm(total=len(xyz_paths), desc="Processing...")
    for _ in pool.imap_unordered(worker, xyz_paths):
        bar.update()
    pool.close(); pool.join(); bar.close()

    # generate splits
    files = sorted(glob.glob(os.path.join(OUT_INT, "*.h5")))
    np.random.shuffle(files)
    n = len(files)
    splits = {"train": files[:int(.8*n)],
              "val":   files[int(.8*n):int(.9*n)],
              "test":  files[int(.9*n):]}
    with open(os.path.join(DST_DIR, "splits.json"), "w") as fp:
        json.dump(splits, fp)
    print(f"Completed processing {n} samples.")

if __name__ == "__main__":
    main()



