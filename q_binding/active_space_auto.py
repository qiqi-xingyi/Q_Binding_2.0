# --*-- conding:utf-8 --*--
# @time:6/29/25 00:04
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:active_space_auto.py

from typing import List, Tuple, Dict

import numpy as np
from pyscf import scf
from qiskit_nature.second_q.transformers import (
    FreezeCoreTransformer,
    ActiveSpaceTransformer,
)

class AutoActiveSpace:
    def __init__(
        self,
        qubit_ceiling: int = 127,
        target_tol: float | None = None,       # kcal/mol; None → single-shot
        occ_thresh: tuple[float, float] = (1.95, 0.05),   # occ / virt cut
    ) -> None:
        self._max_orb = qubit_ceiling // 2
        self._freeze = FreezeCoreTransformer()
        self._tol = target_tol
        self._occ_hi, self._occ_lo = occ_thresh

    # --------------------------------------------------------------
    def from_complex(self, mol_cplx, hf_cplx) -> tuple[list, dict]:
        """
        Returns transformers + metrics after optional ΔE convergence loop.
        """
        mo_occ = hf_cplx.mo_occ
        occ_idx = np.where(mo_occ > self._occ_hi)[0].tolist()
        vir_idx = np.where(mo_occ < self._occ_lo)[0].tolist()

        # start with full occ; enlarging step = +2 virt orbitals
        active = occ_idx.copy()
        step = 2
        last_dE = None

        while True:
            # grow active virtual set if needed
            while len(active) < min(self._max_orb, len(vir_idx)):
                active.append(vir_idx[len(active) - len(occ_idx)])
                if len(active) % step == 0:
                    break

            num_orb = len(active)
            num_elec = 2 * len([i for i in active if i in occ_idx])
            qubits = 2 * num_orb

            act_trf = ActiveSpaceTransformer(
                num_electrons=num_elec,
                num_spatial_orbitals=num_orb,
            )

            # --- break conditions -------------------------------------
            if self._tol is None or last_dE is None:
                # no convergence check yet -> keep expanding or exit later
                e_curr = self._quick_delta_e(mol_cplx, act_trf)
                last_dE = e_curr
            else:
                e_curr = self._quick_delta_e(mol_cplx, act_trf)
                if abs(e_curr - last_dE) < self._tol or qubits >= self._max_orb * 2:
                    break
                last_dE = e_curr

        metrics = {
            "active_orb": num_orb,
            "active_elec": num_elec,
            "qubits": qubits,
        }
        return [self._freeze, act_trf], metrics

    @staticmethod
    def _quick_delta_e(mol_cplx, act_trf) -> float:
        """
        Cheap estimator of binding-energy change for convergence test:
        (HF energy in active space). Returns kcal/mol.
        """
        from qiskit_nature.second_q.drivers import PySCFDriver

        # driver = PySCFDriver(atom=mol_cplx.atom, charge=mol_cplx.charge,
        #                      spin=mol_cplx.spin, basis=mol_cplx.basis)

        atom_strings = [f"{sym} {x:.10f} {y:.10f} {z:.10f}"for sym, (x, y, z) in mol_cplx.atom]
        driver = PySCFDriver(atom=atom_strings, charge=mol_cplx.charge, spin=mol_cplx.spin, basis=mol_cplx.basis)

        problem = driver.run()
        problem = act_trf.transform(problem)

        hf_e = problem.hamiltonian.nuclear_repulsion_energy + \
               problem.reference_energy
        # convert Hartree → kcal/mol
        return hf_e * 627.509