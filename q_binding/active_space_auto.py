# --*-- conding:utf-8 --*--
# @time:6/29/25 00:04
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:active_space_auto.py

from typing import Tuple, Optional
import numpy as np
from qiskit_nature.second_q.transformers import (
    FreezeCoreTransformer,
    ActiveSpaceTransformer,
)


class AutoActiveSpace:
    """
    Lightweight active-space selector for the complex:
      - Avoids running PySCFDriver.run() during active-space sizing;
      - Estimates frozen (N_alpha, N_beta) via an element core-electron table;
      - Ensures the entire spin imbalance (Sz) is carried by the active space,
        so that the inactive space is closed-shell (no 'inactive must be even' error);
      - Uses only the qubit_ceiling as the stopping criterion by default.

    Final Hamiltonians are still built later via PySCFDriver.run() in the main workflow.
    """

    def __init__(
        self,
        qubit_ceiling: int = 127,
        target_tol: Optional[float] = None,         # kept for API compatibility; ignored here
        occ_thresh: Tuple[float, float] = (1.95, 0.05),  # (occ_hi, virt_lo)
    ) -> None:
        self._max_orb = max(1, qubit_ceiling // 2)
        self._freeze = FreezeCoreTransformer()
        self._tol = target_tol  # not used in lightweight mode
        self._occ_hi, self._occ_lo = occ_thresh

    # ------------------------------------------------------------------
    def from_complex(self, mol_cplx, hf_cplx) -> Tuple[list, dict]:
        """
        Decide a unified active-space size on the complex and return:
          [FreezeCoreTransformer(), ActiveSpaceTransformer(...)] and metrics.

        This lightweight version:
          - does NOT call PySCFDriver.run();
          - ignores energy-based convergence; only respects qubit_ceiling.
        """
        # 1) Occupation-based selection bootstrap from complex HF result
        mo_occ = np.atleast_1d(hf_cplx.mo_occ)
        occ_idx = np.where(mo_occ > self._occ_hi)[0].tolist()   # 'deeply occupied' (valence-like)
        vir_idx = np.where(mo_occ < self._occ_lo)[0].tolist()   # 'high virtuals'

        # 2) Determine frozen (N_alpha, N_beta) and Sz using a core-electron table
        na_tot, nb_tot = self._frozen_num_particles(mol_cplx)
        Sz = int(na_tot - nb_tot)  # spin imbalance that MUST be carried by the active space

        # 3) Grow active space from all occupied orbitals, then add virtuals up to ceiling
        active = occ_idx.copy()
        budget = min(self._max_orb, len(occ_idx) + len(vir_idx))
        while len(active) < budget:
            # deterministic pick of next virtual
            next_v = vir_idx[len(active) - len(occ_idx)]
            active.append(next_v)

        num_orb = len(active)              # spatial orbitals in active space
        qubits = 2 * num_orb

        # 4) Build an ActiveSpaceTransformer consistent with Sz
        act_trf, n_act_total = self._build_act_trf_with_sz(num_orb, na_tot, nb_tot, Sz)

        metrics = {
            "active_orb": num_orb,
            "active_elec": n_act_total,    # total active electrons actually used
            "qubits": qubits,
        }
        return [self._freeze, act_trf], metrics

    # ------------------------------------------------------------------
    @staticmethod
    def _core_electrons_for(elem: str) -> int:
        """
        Minimal core-electron table suitable for bio/organic elements.
        Extend if heavier elements are expected.
        """
        elem = elem.capitalize()
        table = {
            # H/He
            "H": 0, "He": 0,
            # 2nd period: freeze 1s^2
            "Li": 2, "Be": 2, "B": 2, "C": 2, "N": 2, "O": 2, "F": 2, "Ne": 2,
            # 3rd period: freeze up to [Ne]
            "Na": 10, "Mg": 10, "Al": 10, "Si": 10, "P": 10, "S": 10, "Cl": 10, "Ar": 10,
        }
        return table.get(elem, 0)

    def _frozen_num_particles(self, mol) -> Tuple[int, int]:
        """
        Return (N_alpha, N_beta) AFTER freezing cores, without running PySCFDriver.

        Rules:
          - Ignore ghost atoms ('GhX' or 'GHOST-X') in electron counting;
          - Subtract per-element core electrons from total electrons;
          - Preserve overall spin S = N_alpha - N_beta = mol.spin.
        """
        from pyscf import gto

        total_e = 0
        frozen_e = 0
        for sym, (x, y, z) in mol.atom:
            if sym.startswith(("Gh", "GHOST-")):
                continue
            elem = sym  # element symbol
            Z = gto.mole.charge(elem)
            total_e += Z
            frozen_e += self._core_electrons_for(elem)

        total_e -= mol.charge
        ne = max(0, total_e - frozen_e)
        S = int(mol.spin)  # N_alpha - N_beta

        # Ensure parity consistency: (ne - S) must be even
        if (ne - S) % 2 != 0:
            # Adjust by one electron if pathological (very rare with sane inputs)
            ne = max(0, ne - 1)

        n_alpha = (ne + S) // 2
        n_beta = ne - n_alpha
        return n_alpha, n_beta

    @staticmethod
    def _build_act_trf_with_sz(
        num_orb: int, na_tot: int, nb_tot: int, Sz: int
    ) -> Tuple[ActiveSpaceTransformer, int]:
        """
        Given the number of active spatial orbitals and the frozen-space
        (N_alpha, N_beta), construct an (n_alpha_act, n_beta_act) tuple so that:
          n_alpha_act - n_beta_act = Sz, and
          n_alpha_act + n_beta_act <= 2 * num_orb,
        with parity (n_act_total - Sz) even.
        """
        max_act_total = min(2 * num_orb, na_tot + nb_tot)
        n_act_total = max_act_total
        if (n_act_total - Sz) % 2 != 0:
            n_act_total -= 1
            if n_act_total < 0:
                n_act_total = 0

        na_act = (n_act_total + Sz) // 2
        nb_act = n_act_total - na_act

        # Clamp to physical ranges (very rare edge cases)
        na_act = max(0, min(na_act, na_tot))
        nb_act = max(0, min(nb_act, nb_tot))
        n_act_total = na_act + nb_act

        act = ActiveSpaceTransformer(
            num_electrons=(na_act, nb_act),
            num_spatial_orbitals=num_orb,
        )
        return act, n_act_total

    # ------------------------------------------------------------------
    @staticmethod
    def _quick_delta_e(mol_cplx) -> float:
        """
        Optional lightweight proxy for energy (unused by default).
        Runs DF-RHF on the complex without generating MO-ERIs to avoid /tmp bloat.
        Returns kcal/mol.
        """
        from pyscf import scf
        mf = scf.RHF(mol_cplx).density_fit()
        mf.conv_tol = 1e-6
        e_hf = mf.kernel()
        return float(e_hf * 627.509)
