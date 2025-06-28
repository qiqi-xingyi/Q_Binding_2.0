# --*-- conding:utf-8 --*--
# @time:6/28/25 17:05
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:main.py

from q_binding import CounterpoiseBuilder


if __name__ == '__main__':

    pdb_path  = "./data/1c5z/1c5z_Binding_mode.pdb"
    plip_path = "./data/1c5z/1c5z_interaction.txt"

    cp = CounterpoiseBuilder(pdb_path, plip_path, ligand_id=("A", "MOL"))

    # Autodetect interacting residues via PLIP, then build CP geometries
    geoms = cp.build_geometries()
    cp.write_xyz("./geom/")         # writes complex.xyz, fragA.xyz, fragB.xyz

    # Obtain PySCF Mole objects if needed
    mole_dict = cp.to_pyscf(basis="def2-SVP")
    energy_complex = mole_dict["complex"].energy_tot()
