# --*-- conding:utf-8 --*--
# @time:6/28/25 17:05
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:main.py

from q_binding import CounterpoiseBuilder
from q_binding import HamiltonianBuilder
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

if __name__ == '__main__':

    pdb = "./data/1c5z/1c5z_Binding_mode.pdb"
    plip = "./data/1c5z/1c5z_interaction.txt"

    cp = CounterpoiseBuilder(pdb, plip, ligand_id=("A", "MOL"))
    cp.build_geometries()
    mole_dict = cp.to_pyscf(basis="def2-SVP")

    ham_builder = HamiltonianBuilder(
        mole_dict,
        transformers=[FreezeCoreTransformer()]  # optional
    )
    ham_ops = ham_builder.build_hamiltonians()
    ham_builder.write_json("./ham/")
