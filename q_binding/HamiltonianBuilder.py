# --*-- conding:utf-8 --*--
# @time:6/28/25 17:56
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:HamiltonianBuilder.py

# hamiltonian.py
from pathlib import Path
from typing import Dict, List, Optional

from qiskit_nature.second_q.drivers import PySCFMoleculeDriver
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import BaseTransformer


class HamiltonianBuilder:
    """
    Convert PySCF Mole objects produced by CounterpoiseBuilder into
    second-quantized Hamiltonians (FermionicOp) via Qiskit Nature.

    Parameters
    ----------
    mole_dict : Dict[str, "pyscf.gto.M"]
        Dictionary mapping tag names ("complex", "fragA", "fragB")
        to PySCF Mole instances.
    basis : str, optional
        Included for signature symmetry; ignored because the basis is already
        defined inside each Mole object.
    transformers : Optional[List[BaseTransformer]]
        Optional list of Qiskit Nature transformer instances
        (e.g. FreezeCoreTransformer, ActiveSpaceTransformer).
    """

    def __init__(
        self,
        mole_dict: Dict[str, "pyscf.gto.M"],
        basis: str = "def2-SVP",
        transformers: Optional[List[BaseTransformer]] = None,
    ):
        self.mole_dict = mole_dict
        self.transformers = transformers or []
        self._hamiltonians: Dict[str, ElectronicEnergy] = {}

    # ------------------------------------------------------------------
    def build_hamiltonians(self) -> Dict[str, ElectronicEnergy]:
        """
        Generate and store Fermionic Hamiltonians for each fragment.

        Returns
        -------
        Dict[str, ElectronicEnergy]
            Mapping tag → second-quantized Hamiltonian (ElectronicEnergy).
        """
        for tag, mole in self.mole_dict.items():
            driver = PySCFMoleculeDriver(mole)
            problem: ElectronicStructureProblem = driver.run()

            for transf in self.transformers:
                problem = transf.transform(problem)

            ham_op: ElectronicEnergy = problem.second_q_ops()[0]
            self._hamiltonians[tag] = ham_op

        return self._hamiltonians

    # ------------------------------------------------------------------
    def write_json(self, out_dir: str = "./ham"):
        """
        Serialize the Hamiltonians to JSON files (one per tag).

        The format follows qiskit-nature's built-in to_json() method.
        """
        if not self._hamiltonians:
            raise RuntimeError("Call build_hamiltonians() before write_json().")

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for tag, ham in self._hamiltonians.items():
            json_path = out_path / f"{tag}.json"
            with json_path.open("w") as fh:
                fh.write(ham.to_json())
