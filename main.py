# --*-- conding:utf-8 --*--
# @time:6/28/25 17:05
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:main.py

from pathlib import Path
from pyscf import scf


from q_binding import CounterpoiseBuilder
from q_binding import HamiltonianBuilder
from q_binding import AutoActiveSpace

from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.drivers import PySCFDriver

HF_TOL = 3
MAX_QB = 10

def main() -> None:

    pdb_path = "./data/1c5z/1c5z_Binding_mode.pdb"
    plip_path = "./data/1c5z/1c5z_interaction.txt"

    # --- CP geometries -------------------------------------------------
    cp = CounterpoiseBuilder(pdb_path, plip_path, ligand_id=("A", "MOL"))
    cp.build_geometries()
    # mole_dict = cp.to_pyscf(basis="def2-SVP")
    mole_dict = cp.to_pyscf(basis="sto-3g")

    # --- HF on complex for active-space decision -----------------------

    auto = AutoActiveSpace(qubit_ceiling=MAX_QB, target_tol=HF_TOL)

    hf_compl = scf.RHF(mole_dict["complex"]).density_fit().run()
    trs, metrics = auto.from_complex(mole_dict["complex"], hf_compl)
    freeze_trf, proto_act = trs
    print(f"active_orb={metrics['active_orb']}, qubits={metrics['qubits']}")

    print(
        f"Active space chosen: {metrics['active_orb']} spatial orbitals, "
        f"{metrics['active_elec']} electrons → {metrics['qubits']} qubits"
    )

    # --- Build Hamiltonians for all three fragments --------------------
    ham_ops = {}
    ham_path = Path("./ham")
    ham_path.mkdir(exist_ok=True)

    for tag in ["complex", "fragA", "fragB"]:

        driver = PySCFDriver(atom=[
            f"{sym} {x} {y} {z}" for sym, (x, y, z) in mole_dict[tag].atom
        ], charge=mole_dict[tag].charge, spin=mole_dict[tag].spin,
            basis=mole_dict[tag].basis)

        problem_raw = driver.run()
        problem_frozen = freeze_trf.transform(problem_raw)


        na, nb = problem_frozen.num_particles
        elec_available = na + nb


        elec_active = min(metrics["active_elec"], elec_available)

        act_trf = ActiveSpaceTransformer(
            num_electrons=elec_active,
            num_spatial_orbitals=metrics["active_orb"],
        )


        hbuilder = HamiltonianBuilder(
            {tag: mole_dict[tag]},
            transformers=[freeze_trf, act_trf]
        )
        op = hbuilder.build_hamiltonians()[tag]
        ham_ops[tag] = op
        (ham_path / f"{tag}.json").write_text(op.to_json())
        print(f"{tag:<7s}: {op.num_alpha + op.num_beta:>3d} e, "
              f"{op.num_spatial_orbitals:>3d} orb → "
              f"{op.num_spatial_orbitals * 2:>3d} qubits")

if __name__ == "__main__":
    main()

