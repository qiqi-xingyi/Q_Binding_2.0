# --*-- conding:utf-8 --*--
# @time:8/13/25 17:29
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:creat_data.py

# scripts/batch_cp.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

from pyscf import scf
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from q_binding import CounterpoiseBuilder, AutoActiveSpace, HamiltonianBuilder


@dataclass
class CPConfig:
    basis: str = "def2-SVP"                 # e.g. "def2-SVP"
    qubit_ceiling: int = 96               # total qubit budget
    target_tol: Optional[float] = 0.5     # kcal/mol; None → single-shot
    occ_thresh: Tuple[float, float] = (1.95, 0.05)  # (occ_hi, occ_lo)
    overwrite: bool = True


class BenchmarkProcessor:
    """
    Batch processor for CP geometries and aligned electronic integrals.

    For each case:
      1) Build CP geometries (complex/fragA/fragB) with GHOST- labels.
      2) Choose a unified active space on the complex (freeze core + trimming).
      3) For each fragment, compute (n_alpha, n_beta) after freezing and
         build the ActiveSpaceTransformer with the SAME number of orbitals.
      4) Export Qiskit-Nature JSON (ElectronicEnergy) per fragment.
    """

    def __init__(self, cfg: CPConfig):
        self.cfg = cfg

    # --------------------------- public API ---------------------------

    def run_all(self, root: Path) -> None:
        """
        Discover cases under `root` and process them.
        A case directory must contain one PDB and one PLIP interaction file.
        """
        pairs = self._discover_pairs(root)
        if not pairs:
            print(f"[WARN] No cases found under: {root}")
            return
        print(f"[INFO] Found {len(pairs)} cases.")
        for pdb_path, plip_path in pairs:
            self.run_case(pdb_path, plip_path)

    def run_case(self, pdb_path: Path, plip_path: Path) -> None:
        case_dir = pdb_path.parent
        out_geom = case_dir / "geom"
        out_ham = case_dir / "ham"
        out_geom.mkdir(parents=True, exist_ok=True)
        out_ham.mkdir(parents=True, exist_ok=True)

        # 1) Build CP geometries
        cp = CounterpoiseBuilder(str(pdb_path), str(plip_path), ligand_id=("A", "MOL"))
        cp.build_geometries()
        cp.write_xyz(out_geom.as_posix())

        # 2) To PySCF Mole (ghost atoms supported)
        mole_dict = cp.to_pyscf(basis=self.cfg.basis)

        # 3) Decide active space on the complex with HF occupations
        hf_compl = scf.RHF(mole_dict["complex"]).density_fit().run()
        auto = AutoActiveSpace(
            qubit_ceiling=self.cfg.qubit_ceiling,
            target_tol=self.cfg.target_tol,
            occ_thresh=self.cfg.occ_thresh,
        )
        transformers, metrics = auto.from_complex(mole_dict["complex"], hf_compl)
        freeze_trf, _proto_act = transformers

        # 4) Build per-fragment Hamiltonians with aligned active-space size
        ham_ops: Dict[str, object] = {}
        hf_check: Dict[str, float] = {}

        for tag in ["complex", "fragA", "fragB"]:
            # 4a) Build raw problem through driver
            atom_strings = [
                f"{sym} {x:.10f} {y:.10f} {z:.10f}"
                for sym, (x, y, z) in mole_dict[tag].atom
            ]
            driver = PySCFDriver(
                atom=atom_strings,
                charge=mole_dict[tag].charge,
                spin=mole_dict[tag].spin,
                basis=mole_dict[tag].basis,
            )
            problem_raw = driver.run()
            problem_frozen = freeze_trf.transform(problem_raw)

            # 4b) Read (n_alpha, n_beta) after freezing; tuple is required for open-shell
            n_alpha, n_beta = problem_frozen.num_particles
            elec_tuple = (n_alpha, n_beta)

            act_trf = ActiveSpaceTransformer(
                num_electrons=elec_tuple,
                num_spatial_orbitals=metrics["active_orb"],
            )

            # 4c) Build Hamiltonian with the SAME freeze + per-fragment electrons
            hbuilder = HamiltonianBuilder({tag: mole_dict[tag]}, transformers=[freeze_trf, act_trf])
            op = hbuilder.build_hamiltonians()[tag]
            ham_ops[tag] = op

            # 4d) Persist JSON
            (out_ham / f"{tag}.json").write_text(op.to_json())

            # 4e) Quick HF reference for sanity check (optional)
            hf_check[tag] = float(
                problem_frozen.hamiltonian.nuclear_repulsion_energy + problem_frozen.reference_energy
            )

        # 5) Write checkpoint / metrics
        info = {
            "basis": self.cfg.basis,
            "active_orb": metrics["active_orb"],
            "qubits": metrics["active_orb"] * 2,
            "case": case_dir.name,
            "hf_reference": hf_check,
        }
        (case_dir / "checkpoint.txt").write_text(json.dumps(info, indent=2))
        print(
            f"[DONE] {case_dir.name}: {metrics['active_orb']} orb → {metrics['active_orb']*2} qubits | "
            f"JSON at {out_ham}"
        )

    # --------------------------- helpers ---------------------------

    def _discover_pairs(self, root: Path) -> List[Tuple[Path, Path]]:
        """
        Find (pdb_path, plip_path) pairs under given root.
        Assumes naming like *_Binding_mode.pdb and *_interaction.txt, or just 1 PDB + 1 TXT per leaf folder.
        """
        pairs: List[Tuple[Path, Path]] = []
        for case in sorted(p for p in root.glob("**/") if p.is_dir()):
            pdbs = list(case.glob("*Binding_mode.pdb")) or list(case.glob("*.pdb"))
            plips = list(case.glob("*interaction.txt")) or list(case.glob("*.txt"))
            if len(pdbs) == 1 and len(plips) == 1:
                pairs.append((pdbs[0], plips[0]))
        return pairs


if __name__ == "__main__":
    # Example usage:
    #   python scripts/batch_cp.py /path/to/data_set
    import sys
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data_set")
    cfg = CPConfig(
        basis="sto-3g",            # change to "def2-SVP" for production
        qubit_ceiling=96,          # total qubit cap (2 * active_orb)
        target_tol=0.5,            # kcal/mol; None to skip iterative growth
        occ_thresh=(1.95, 0.05),
        overwrite=True,
    )
    BenchmarkProcessor(cfg).run_all(data_root)
