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
    """
    Build a transformer list that freezes core electrons and trims the
    active space so total qubits <= qubit_budget (default: 127).
    The same active-orbital list must be reused for all CP fragments.
    """

    def __init__(self, qubit_budget: int = 127) -> None:
        self._max_orb = qubit_budget // 2
        self._freeze = FreezeCoreTransformer()

    # --------------------------------------------------------------
    def from_complex(
        self, mol_complex, hf_complex: "scf.hf.RHF"
    ) -> Tuple[List, Dict]:
        """
        Decide an active space on the complex fragment, then return:
            • transformer list  [FreezeCore, ActiveSpace]
            • metrics dict  {total_orb, active_orb, active_elec, qubits}
        """
        mo_occ = hf_complex.mo_occ
        norb_total = len(mo_occ)

        occ_idx = np.where(mo_occ > 1.95)[0].tolist()   # valence-like
        vir_idx = np.where(mo_occ < 0.05)[0].tolist()   # high-energy virtual

        active_orb = occ_idx.copy()
        for idx in vir_idx:
            if len(active_orb) >= self._max_orb:
                break
            active_orb.append(idx)

        n_active_occ = len(
            [i for i in active_orb if i in occ_idx]
        )
        num_elec = 2 * n_active_occ
        num_orb = len(active_orb)

        act_trf = ActiveSpaceTransformer(
            num_electrons=num_elec,
            num_spatial_orbitals=num_orb,
            active_orbitals=active_orb,
        )

        metrics = {
            "total_orb": norb_total,
            "active_orb": num_orb,
            "active_elec": num_elec,
            "qubits": 2 * num_orb,
        }
        return [self._freeze, act_trf], metrics