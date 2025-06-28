# --*-- conding:utf-8 --*--
# @time:6/28/25 17:05
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:main.py

# main.py
from pathlib import Path

from q_binding import CounterpoiseBuilder
from q_binding import HamiltonianBuilder


def main() -> None:

    pdb_path = "./data/1c5z/1c5z_Binding_mode.pdb"
    plip_path = "./data/1c5z/1c5z_interaction.txt"


    cp = CounterpoiseBuilder(pdb_path, plip_path, ligand_id=("A", "MOL"))

    # Build geometries with PLIP-detected residues (default mode)
    cp.build_geometries()

    # Optional: write XYZ files for inspection
    cp.write_xyz("./geom")

    mole_dict = cp.to_pyscf(basis="def2-SVP")


    ham_builder = HamiltonianBuilder(mole_dict)
    ham_ops = ham_builder.build_hamiltonians()

    # Persist each Hamiltonian to JSON for later VQE runs
    ham_out = Path("./ham")
    ham_out.mkdir(exist_ok=True)
    ham_builder.write_json(ham_out.as_posix())

    # for tag, ham in ham_ops.items():
    #     nelec = ham.num_alpha + ham.num_beta
    #     norb = ham.num_spatial_orbitals
    #     print(f"{tag:<8s}: {nelec:>3d} electrons, {norb:>3d} spatial orbitals")

if __name__ == "__main__":
    main()

