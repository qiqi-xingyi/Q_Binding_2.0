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


def main() -> None:
    pdb_path = "./data/1c5z/1c5z_Binding_mode.pdb"
    plip_path = "./data/1c5z/1c5z_interaction.txt"

    # --- CP geometries -------------------------------------------------
    cp = CounterpoiseBuilder(pdb_path, plip_path, ligand_id=("A", "MOL"))
    cp.build_geometries()
    mole_dict = cp.to_pyscf(basis="def2-SVP")

    # --- HF on complex for active-space decision -----------------------
    hf_compl = scf.RHF(mole_dict["complex"]).run()
    auto = AutoActiveSpace(qubit_budget=127)
    trfs, metrics = auto.from_complex(mole_dict["complex"], hf_compl)

    print(
        f"Active space chosen: {metrics['active_orb']} spatial orbitals, "
        f"{metrics['active_elec']} electrons → {metrics['qubits']} qubits"
    )

    # --- Build Hamiltonians for all three fragments --------------------
    ham_ops = {}
    ham_path = Path("./ham")
    ham_path.mkdir(exist_ok=True)

    for tag in ["complex", "fragA", "fragB"]:
        hbuilder = HamiltonianBuilder(
            {tag: mole_dict[tag]}, transformers=trfs
        )
        op = hbuilder.build_hamiltonians()[tag]
        ham_ops[tag] = op
        # persist
        (ham_path / f"{tag}.json").write_text(op.to_json())
        print(f"{tag:<7s}: {op.num_alpha + op.num_beta:>3d} e, "
              f"{op.num_spatial_orbitals:>3d} orb → "
              f"{op.num_spatial_orbitals*2:>3d} qubits")

if __name__ == "__main__":
    main()

