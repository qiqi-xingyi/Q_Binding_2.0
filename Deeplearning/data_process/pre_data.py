# --*-- conding:utf-8 --*--
# @time:7/29/25 11:06
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:pre_data.py

import os, glob, json, argparse, multiprocessing as mp
import numpy as np, h5py
from tqdm import tqdm
from pyscf import gto, scf, ao2mo
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems      import ElectronicStructureProblem
from qiskit_nature.second_q.converters import QubitConverter
from qiskit_nature.second_q.mappers    import JordanWignerMapper


def read_xyz(path):
    with open(path) as fh:
        lines = fh.readlines()
    n = int(lines[0]); Z, C = [], []
    for ln in lines[2:2+n]:
        sym, x, y, z = ln.split()[:4]
        Z.append(gto.mole._charge(sym))
        C.append([float(x), float(y), float(z)])
    return np.array(Z, np.int16), np.array(C, np.float32)

def indep_mask(n):
    idx = np.triu_indices(n)
    mask = np.zeros((n,n,n,n), bool)
    for p,q in zip(*idx):
        for r,s in zip(*idx):
            mask[p,q,r,s] = (p*n+q) >= (r*n+s)
    return mask.reshape(-1)

mask_cache = {}
def integrals(Z, C, basis):
    mol = gto.M(atom=list(zip(Z, C)), basis=basis, unit="Angstrom", cart=False)
    n = mol.nao_nr()
    mf = scf.RHF(mol).run()
    vec_h = mf.get_hcore()[np.triu_indices(n)].astype("float32")
    eri   = ao2mo.restore(1, mol.intor('int2e'), n).reshape(-1)
    if n not in mask_cache:
        mask_cache[n] = indep_mask(n)
    vec_g = eri[mask_cache[n]].astype("float32")
    return vec_h, vec_g, n

def build_pauli(h_vec, g_vec, n, eps):

    h = np.zeros((n,n)); h[np.triu_indices(n)] = h_vec
    h = h + h.T - np.diag(np.diag(h))
    g = np.zeros((n,n,n,n)); g.reshape(-1)[mask_cache[n]] = g_vec
    g = (g + np.einsum('pqrs->qpsr', g) + np.einsum('pqrs->rspq', g)) / 3

    elH = ElectronicEnergy.from_raw_integrals(h, g, n)
    prob = ElectronicStructureProblem(elH)
    op   = QubitConverter(JordanWignerMapper()).convert(
                prob.second_q_ops()["ElectronicEnergy"])

    strings, coeffs = [], []
    for P, c in zip(op.paulis, op.coeffs.real):
        if abs(c) > eps:
            strings.append(P.to_label())
            coeffs.append(np.float32(c))
    return np.array(strings, np.bytes_), np.array(coeffs, np.float32)

# ---------- Worker ----------
def process(args):
    xyz_path, out_int, out_pau, basis, eps = args
    stem = os.path.splitext(os.path.basename(xyz_path))[0]   # gdb1
    Z, C = read_xyz(xyz_path)
    h, g, n = integrals(Z, C, basis)
    strs, coeffs = build_pauli(h, g, n, eps)

    # 1) 积分文件
    with h5py.File(f"{out_int}/{stem}.h5", "w") as f:
        f["Z"]=Z; f["pos"]=C; f["h"]=h; f["g"]=g
        f.attrs.update(n_orb=n, basis=basis)

    # 2) Pauli 文件
    with h5py.File(f"{out_pau}/{stem}.h5", "w") as f:
        f["strings"]=strs; f["coeffs"]=coeffs
        f.attrs.update(mapper="JW", eps=eps, n_orb=n, basis=basis)
    return stem

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",  required=True, help="dir of *.xyz")
    ap.add_argument("--dst",  default="data", help="parent output dir")
    ap.add_argument("--basis",default="sto3g")
    ap.add_argument("--eps",  type=float, default=1e-4)
    ap.add_argument("--n_proc",type=int, default=8)
    args = ap.parse_args()

    out_int = f"{args.dst}/integrals_h5"
    out_pau = f"{args.dst}/paulis_h5"
    os.makedirs(out_int, exist_ok=True)
    os.makedirs(out_pau, exist_ok=True)

    xyz_files = sorted(glob.glob(f"{args.src}/*.xyz"))
    pool = mp.Pool(args.n_proc)
    bar  = tqdm(total=len(xyz_files), desc="QM9→(h,g)+Pauli")
    for _ in pool.imap_unordered(lambda p: process((p,out_int,out_pau,
                                                    args.basis,args.eps)),
                                 xyz_files):
        bar.update()
    pool.close(); pool.join(); bar.close()


    files = sorted(glob.glob(f"{out_int}/*.h5"))
    import random; random.shuffle(files)
    n = len(files)
    splits = {"train": files[:int(.8*n)],
              "val":   files[int(.8*n):int(.9*n)],
              "test":  files[int(.9*n):]}
    json.dump(splits, open(f"{args.dst}/splits.json","w"))
    print("Done:", n, "samples")

if __name__ == "__main__":
    main()
