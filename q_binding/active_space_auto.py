# --*-- conding:utf-8 --*--
# @time:6/29/25 00:04
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:active_space_auto.py

# q_binding/active_space_auto.py
from typing import List, Tuple, Dict, Optional

import numpy as np
# PySCF SCF object may be passed from caller; we do not run SCF here.
from qiskit_nature.second_q.transformers import (
    FreezeCoreTransformer,
    ActiveSpaceTransformer,
)


class AutoActiveSpace:
    """
    Decide a unified freeze-core + active-space on the complex, then reuse the
    same number of spatial orbitals for all fragments. The active-electron tuple
    (n_alpha, n_beta) is constructed so that the entire spin imbalance Sz is
    carried by the active space, guaranteeing a closed-shell inactive space and
    avoiding Qiskit Nature's 'inactive electrons must be even' error.
    """

    def __init__(
        self,
        qubit_ceiling: int = 127,
        target_tol: Optional[float] = None,         # kcal/mol; None → single-shot
        occ_thresh: Tuple[float, float] = (1.95, 0.05),  # (occ_hi, virt_lo)
    ) -> None:
        self._max_orb = qubit_ceiling // 2
        self._freeze = FreezeCoreTransformer()
        self._tol = target_tol
        self._occ_hi, self._occ_lo = occ_thresh

    # ------------------------------------------------------------------
    def from_complex(self, mol_cplx, hf_cplx) -> Tuple[list, dict]:
        """
        Build transformers on the complex:
          - FreezeCoreTransformer (shared across all fragments)
          - ActiveSpaceTransformer with a size decided here, and with
            (n_alpha, n_beta) carrying the full frozen Sz.
        Returns ([freeze, active], metrics).
        """
        # HF occupations on the complex (caller provides RHF/UHF results)
        mo_occ = hf_cplx.mo_occ
        occ_idx = np.where(mo_occ > self._occ_hi)[0].tolist()
        vir_idx = np.where(mo_occ < self._occ_lo)[0].tolist()

        # Determine Sz after freezing cores ON THE COMPLEX.
        na_tot, nb_tot = self._frozen_num_particles(mol_cplx)
        Sz = int(na_tot - nb_tot)  # spin imbalance to be carried by active space

        # Start from all occupied; enlarge by adding virtuals in steps of 2
        active = occ_idx.copy()
        step = 2
        last_dE = None

        # Helper to build an ActiveSpaceTransformer consistent with Sz
        def build_act_trf(num_orb: int) -> Tuple[ActiveSpaceTransformer, int]:
            # Max electrons we can put into active space is bounded by both
            # available electrons and the number of orbitals.
            max_act_total = min(2 * num_orb, na_tot + nb_tot)

            # Ensure parity consistency with Sz: (n_act_total - Sz) must be even.
            n_act_total = max_act_total
            if (n_act_total - Sz) % 2 != 0:
                # prefer decreasing by 1 (to stay within 2*num_orb)
                n_act_total -= 1
                if n_act_total < 0:
                    n_act_total = 0

            # Distribute to spins so that active carries all Sz.
            na_act = (n_act_total + Sz) // 2
            nb_act = n_act_total - na_act

            # Clamp to physical ranges (rare edge-cases).
            na_act = max(0, min(na_act, na_tot))
            nb_act = max(0, min(nb_act, nb_tot))
            n_act_total = na_act + nb_act  # refresh after clamping

            act = ActiveSpaceTransformer(
                num_electrons=(na_act, nb_act),
                num_spatial_orbitals=num_orb,
            )
            return act, n_act_total

        while True:
            # Grow active virtuals only if we still have budget and virtuals left.
            while len(active) < min(self._max_orb, len(vir_idx)):
                # pick next virtual in a deterministic order
                next_v = vir_idx[len(active) - len(occ_idx)]
                active.append(next_v)
                if len(active) % step == 0:
                    break

            num_orb = len(active)
            qubits = 2 * num_orb

            act_trf, n_act_total = build_act_trf(num_orb)

            # Convergence / budget checks using a cheap HF-in-active estimator
            e_curr = self._quick_delta_e(mol_cplx, num_orb)
            if self._tol is None:
                # Single-shot: stop when we hit ceiling or no more virtuals added
                if qubits >= self._max_orb * 2 or len(active) >= len(occ_idx) + len(vir_idx):
                    break
            else:
                if last_dE is not None and (abs(e_curr - last_dE) < self._tol or qubits >= self._max_orb * 2):
                    break
                last_dE = e_curr

        metrics = {
            "active_orb": num_orb,
            "active_elec": n_act_total,   # total active electrons actually used
            "qubits": qubits,
        }
        return [self._freeze, act_trf], metrics

    # ------------------------------------------------------------------
    def _frozen_num_particles(self, mol) -> Tuple[int, int]:
        """Return (N_alpha, N_beta) after applying FreezeCore to 'mol'."""
        from qiskit_nature.second_q.drivers import PySCFDriver

        atom_strings = [f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in mol.atom]
        driver = PySCFDriver(
            atom=atom_strings,
            charge=mol.charge,
            spin=mol.spin,
            basis=mol.basis,
        )
        problem = driver.run()
        problem_frozen = self._freeze.transform(problem)
        return problem_frozen.num_particles  # (N_alpha, N_beta)

    # ------------------------------------------------------------------
    def _quick_delta_e(self, mol_cplx, num_orb: int) -> float:
        """
        Cheap estimator (HF-in-active) used only for convergence of the
        active-space size. We ensure inactive space is closed-shell by:
          1) Freeze cores;
          2) Put the full Sz into the active-electron tuple.
        Returns kcal/mol.
        """
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, FreezeCoreTransformer

        # Build raw problem
        atom_strings = [f"{sym} {x:.10f} {y:.10f} {z:.10f}" for sym, (x, y, z) in mol_cplx.atom]
        driver = PySCFDriver(
            atom=atom_strings,
            charge=mol_cplx.charge,
            spin=mol_cplx.spin,
            basis=mol_cplx.basis,
        )
        problem = driver.run()

        # Freeze cores first (use a local transformer to keep this method self-contained)
        freeze = FreezeCoreTransformer()
        problem_frozen = freeze.transform(problem)
        na_tot, nb_tot = problem_frozen.num_particles
        Sz = int(na_tot - nb_tot)

        # Build an active-space consistent with Sz and the given num_orb
        max_act_total = min(2 * num_orb, na_tot + nb_tot)
        n_act_total = max_act_total
        if (n_act_total - Sz) % 2 != 0:
            n_act_total -= 1
            if n_act_total < 0:
                n_act_total = 0
        na_act = (n_act_total + Sz) // 2
        nb_act = n_act_total - na_act

        act_trf = ActiveSpaceTransformer(
            num_electrons=(na_act, nb_act),
            num_spatial_orbitals=num_orb,
        )
        problem_act = act_trf.transform(problem_frozen)

        # HF reference energy (Hartree) → kcal/mol
        # hf_e = (
        #     problem_act.hamiltonian.nuclear_repulsion_energy
        #     + problem_act.reference_energy
        # )
        # return float(hf_e * 627.509)

        # HF reference energy (Hartree) → kcal/mol  (version-safe access)
        # Try modern locations first, then fall back gracefully.

        e_nuc = None
        try:

            e_nuc = float(problem_act.hamiltonian.nuclear_repulsion_energy)
        except Exception:

            e_nuc = float(getattr(problem_act.hamiltonian, "_nuclear_repulsion_energy", 0.0))

        e_ref = None

        if hasattr(problem_act, "reference_energy") and problem_act.reference_energy is not None:
            e_ref = float(problem_act.reference_energy)

        elif getattr(problem_act, "properties", None) is not None:
            ee_prop = getattr(problem_act.properties, "electronic_energy", None)
            if ee_prop is not None and getattr(ee_prop, "reference_energy", None) is not None:
                e_ref = float(ee_prop.reference_energy)

        if e_ref is None:
            e_ref = 0.0

        hf_e = e_nuc + e_ref
        return float(hf_e * 627.509)

