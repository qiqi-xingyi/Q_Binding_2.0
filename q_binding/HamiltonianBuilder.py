# --*-- conding:utf-8 --*--
# @time:6/28/25 17:56
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:HamiltonianBuilder.py

# q_binding/hamiltonian.py
from pathlib import Path
from typing import Dict, List, Optional

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import BaseTransformer


class HamiltonianBuilder:
    """
    Build second-quantized Hamiltonians (ElectronicEnergy) from
    PySCF Mole objects produced by CounterpoiseBuilder.
    """

    def __init__(
        self,
        mole_dict: Dict[str, "pyscf.gto.Mole"],
        transformers: Optional[List[BaseTransformer]] = None,
    ):
        self.mole_dict = mole_dict
        self.transformers = transformers or []
        self._ham_ops: Dict[str, ElectronicEnergy] = {}

    # ------------------------------------------------------------------
    def build_hamiltonians(self) -> Dict[str, ElectronicEnergy]:
        """Generate Hamiltonians for all fragments."""
        for tag, mol in self.mole_dict.items():
            atom_list = [
                f"{sym} {x:.10f} {y:.10f} {z:.10f}"
                for sym, (x, y, z) in mol.atom
            ]

            driver = PySCFDriver(
                atom=atom_list,
                charge=mol.charge,
                spin=mol.spin,
                basis=mol.basis,
            )

            problem: ElectronicStructureProblem = driver.run()

            for tr in self.transformers:
                problem = tr.transform(problem)

            ham_op: ElectronicEnergy = problem.second_q_ops()[0]
            self._ham_ops[tag] = ham_op

        return self._ham_ops

    # ------------------------------------------------------------------
    def write_json(self, out_dir: str = "./ham"):
        """Save each Hamiltonian as a JSON file."""
        if not self._ham_ops:
            raise RuntimeError("Call build_hamiltonians() first.")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for tag, ham in self._ham_ops.items():
            (Path(out_dir) / f"{tag}.json").write_text(ham.to_json())

