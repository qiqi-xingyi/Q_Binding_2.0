# --*-- conding:utf-8 --*--
# @time:8/13/25 17:29
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:create_data.py

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict

from pyscf import scf
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from q_binding import CounterpoiseBuilder, AutoActiveSpace, HamiltonianBuilder

@dataclass
class CPConfig:
    # basis: str = "def2-SVP"
    basis: str = "sto-3g"
    qubit_ceiling: int = 127
    target_tol: Optional[float] = 0.5
    occ_thresh: Tuple[float, float] = (1.95, 0.05)
    overwrite: bool = True

class BenchmarkProcessor:
    def __init__(self, cfg: CPConfig):
        self.cfg = cfg

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

        cp = CounterpoiseBuilder(str(pdb_path), str(plip_path), ligand_id=("A", "MOL"))
        cp.build_geometries()
        cp.write_xyz(out_geom.as_posix())

        mole_dict = cp.to_pyscf(basis=self.cfg.basis)

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
            n_alpha, n_beta = problem_frozen.num_particles

            act_trf = ActiveSpaceTransformer(
                num_electrons=(n_alpha, n_beta),
                num_spatial_orbitals=metrics["active_orb"],
            )

            hbuilder = HamiltonianBuilder({tag: mole_dict[tag]}, transformers=[freeze_trf, act_trf])
            op = hbuilder.build_hamiltonians()[tag]
            (out_ham / f"{tag}.json").write_text(op.to_json())


            hf_ref[tag] = float(
                problem_frozen.hamiltonian.nuclear_repulsion_energy + problem_frozen.reference_energy
            )
            print(f"[{case_dir.name}] {tag:<7s}: {op.num_spatial_orbitals:>3d} orb → {op.num_spatial_orbitals*2:>3d} qubits")

        info = {
            "basis": self.cfg.basis,
            "active_orb": metrics["active_orb"],
            "qubits": metrics["active_orb"] * 2,
            "case": case_dir.name,
            "hf_reference": hf_ref,
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


    DATA_ROOT = Path("./data/1c5z")
    cfg = CPConfig(
        # basis="def2-SVP",
        basis="sto-3g",
        qubit_ceiling=127,
        target_tol=0.5,
        occ_thresh=(1.95, 0.05),
        overwrite=True,
    )
    BenchmarkProcessor(cfg).run_all(DATA_ROOT)

