# --*-- conding:utf-8 --*--
# @time:8/13/25 17:29
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:create_data.py

# create_data.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict

from pyscf import scf
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import SparsePauliOp

from q_binding import CounterpoiseBuilder, AutoActiveSpace, HamiltonianBuilder

from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp
import json
import numpy as np


# ---------- helpers: serialization ----------
def _pairs_from_sparse_label_op(op) -> list[tuple[str, complex]]:
    """Version-robust extraction of (label, coeff) pairs."""
    if hasattr(op, "to_list"):  # preferred
        return [(lbl, complex(c)) for lbl, c in op.to_list()]
    if hasattr(op, "to_sparse_list"):  # older fall-back
        return [(lbl, complex(c)) for lbl, c in op.to_sparse_list()]
    # last resort: labels/coeffs arrays
    if hasattr(op, "labels") and hasattr(op, "coeffs"):
        labels = list(op.labels)
        coeffs = np.array(op.coeffs, dtype=complex).tolist()
        return list(zip(labels, map(complex, coeffs)))
    raise TypeError(f"Unsupported operator type for serialization: {type(op)}")

def fermionic_to_json(op: FermionicOp) -> str:
    data = [(label, [c.real, c.imag]) for label, c in _pairs_from_sparse_label_op(op)]
    return json.dumps({"type": "FermionicOp", "version": "0.7+", "data": data})

def sparse_pauli_to_json(op: SparsePauliOp) -> str:
    data = [(label, [c.real, c.imag]) for label, c in _pairs_from_sparse_label_op(op)]
    return json.dumps({"type": "SparsePauliOp", "version": "1.0+", "data": data})

# ---------------- config ----------------
@dataclass
class CPConfig:
    basis: str = "sto-3g"  # use "def2-SVP" for higher accuracy on selected cases
    qubit_ceiling: int = 127
    target_tol: Optional[float] = 0.5
    occ_thresh: Tuple[float, float] = (1.95, 0.05)
    overwrite: bool = True


# -------------- processor ---------------
class BenchmarkProcessor:
    def __init__(self, cfg: CPConfig):
        self.cfg = cfg
        self._mapper = JordanWignerMapper()  # fixed JW mapping

    def run_all(self, root: Path) -> None:
        pairs = self._discover_pairs(root)
        print(f"[INFO] Found {len(pairs)} cases under: {root}")
        for pdb_path, plip_path in pairs:
            self.run_case(pdb_path, plip_path)

    def run_case(self, pdb_path: Path, plip_path: Path) -> None:
        case_dir = pdb_path.parent
        out_geom = case_dir / "geom"
        out_ham = case_dir / "ham"
        out_geom.mkdir(parents=True, exist_ok=True)
        out_ham.mkdir(parents=True, exist_ok=True)

        # CP geometries
        cp = CounterpoiseBuilder(str(pdb_path), str(plip_path), ligand_id=("A", "MOL"))
        cp.build_geometries()
        cp.write_xyz(out_geom.as_posix())
        mole_dict = cp.to_pyscf(basis=self.cfg.basis)

        # Active-space decision on complex
        hf_compl = scf.RHF(mole_dict["complex"]).density_fit().run()
        auto = AutoActiveSpace(
            qubit_ceiling=self.cfg.qubit_ceiling,
            target_tol=self.cfg.target_tol,
            occ_thresh=self.cfg.occ_thresh,
        )
        transformers, metrics = auto.from_complex(mole_dict["complex"], hf_compl)
        freeze_trf, _ = transformers

        hf_ref: Dict[str, float] = {}
        for tag in ["complex", "fragA", "fragB"]:
            atom_strings = [f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in mole_dict[tag].atom]
            driver = PySCFDriver(
                atom=atom_strings,
                charge=mole_dict[tag].charge,
                spin=mole_dict[tag].spin,
                basis=mole_dict[tag].basis,
            )
            problem_raw = driver.run()
            problem_frozen = freeze_trf.transform(problem_raw)

            # fragment-specific active electrons consistent with Sz and capacity
            n_alpha, n_beta = problem_frozen.num_particles
            Sz = n_alpha - n_beta
            num_orb = metrics["active_orb"]
            max_act_total = min(2 * num_orb, n_alpha + n_beta)
            n_act_total = max_act_total
            if (n_act_total - Sz) % 2 != 0:
                n_act_total -= 1
                if n_act_total < 0:
                    n_act_total = 0
            na_act = (n_act_total + Sz) // 2
            nb_act = n_act_total - na_act
            na_act = max(0, min(na_act, n_alpha))
            nb_act = max(0, min(nb_act, n_beta))

            act_trf = ActiveSpaceTransformer(
                num_electrons=(na_act, nb_act),
                num_spatial_orbitals=num_orb,
            )

            # --- build ElectronicEnergy, then get FermionicOp ---
            e_energy = HamiltonianBuilder(
                {tag: mole_dict[tag]}, transformers=[freeze_trf, act_trf]
            ).build_hamiltonians()[tag]                  # ElectronicEnergy
            f_op: FermionicOp = e_energy.second_q_op()   # FermionicOp

            # Save FermionicOp
            (out_ham / f"{tag}.json").write_text(fermionic_to_json(f_op))

            # Map to Pauli with JW and save
            q_op: SparsePauliOp = self._mapper.map(f_op)
            (out_ham / f"{tag}.pauli.json").write_text(sparse_pauli_to_json(q_op))

            # version-tolerant HF reference (log only)
            try:
                e_nuc = float(problem_frozen.hamiltonian.nuclear_repulsion_energy)
            except Exception:
                e_nuc = float(getattr(problem_frozen.hamiltonian, "_nuclear_repulsion_energy", 0.0))
            e_ref = None
            if hasattr(problem_frozen, "reference_energy") and problem_frozen.reference_energy is not None:
                e_ref = float(problem_frozen.reference_energy)
            elif getattr(problem_frozen, "properties", None) is not None:
                ee_prop = getattr(problem_frozen.properties, "electronic_energy", None)
                if ee_prop is not None and getattr(ee_prop, "reference_energy", None) is not None:
                    e_ref = float(ee_prop.reference_energy)
            if e_ref is None:
                e_ref = 0.0
            hf_ref[tag] = e_nuc + e_ref

            # dimensions
            try:
                nspin = f_op.register_length
                norb = nspin // 2
            except Exception:
                norb = metrics["active_orb"]
                nspin = 2 * norb
            print(f"[{case_dir.name}] {tag:<7s}: {norb:>3d} orb → {nspin:>3d} qubits (JW, {q_op.num_qubits} qb)")

        info = {
            "basis": self.cfg.basis,
            "active_orb": metrics["active_orb"],
            "qubits": metrics["active_orb"] * 2,
            "case": case_dir.name,
            "hf_reference": hf_ref,
            "mapper": "jordan-wigner",
        }
        (case_dir / "checkpoint.txt").write_text(json.dumps(info, indent=2))
        print(f"[DONE] {case_dir.name} → {out_ham}")

    def _discover_pairs(self, root: Path):
        pairs = []
        for case in sorted(p for p in root.glob("**/") if p.is_dir()):
            pdbs = list(case.glob("*Binding_mode.pdb")) or list(case.glob("*.pdb"))
            plips = list(case.glob("*interaction.txt")) or list(case.glob("*.txt"))
            if len(pdbs) == 1 and len(plips) == 1:
                pairs.append((pdbs[0], plips[0]))
        return pairs


if __name__ == "__main__":
    DATA_ROOT = Path("./data/benchmark_binidng_sites")
    cfg = CPConfig(
        basis="sto-3g",
        qubit_ceiling=127,
        target_tol=0.5,
        occ_thresh=(1.95, 0.05),
        overwrite=True,
    )
    BenchmarkProcessor(cfg).run_all(DATA_ROOT)


