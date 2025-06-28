# --*-- conding:utf-8 --*--
# @time:6/28/25 17:04
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:CounterpoiseBuilder.py

# counterpoise.py
from pathlib import Path
import re
from typing import List, Dict, Tuple

class CounterpoiseBuilder:
    """
    Build XYZ geometries needed for counterpoise (CP) correction
    from a PDB structure and its PLIP interaction report.

    Output:
        • complex.xyz  : real atoms of both ligand and protein
        • fragA_cp.xyz : ligand real atoms + protein ghost atoms
        • fragB_cp.xyz : protein real atoms (selected residues) + ligand ghost atoms
    The class also exposes a PySCF‐ready Mole dictionary.
    """
    _pdb_atom_line = re.compile(r"^(ATOM|HETATM)")

    def __init__(
        self,
        pdb_path: str,
        plip_path: str,
        ligand_id: Tuple[str, str] = ("A", "MOL"),
    ):
        self.pdb_path = Path(pdb_path)
        self.plip_path = Path(plip_path)
        self.lig_chain, self.lig_resname = ligand_id
        self.atoms = self._parse_pdb()
        self.hot_residue_ids = self._parse_plip()  # e.g. ['SER:B:190', ...]

    # ---------- public API ----------
    def build_geometries(
        self, residue_scope: str | List[str] = "auto"
    ) -> Dict[str, str]:
        """
        Return a dict {'complex': xyz, 'fragA': xyz, 'fragB': xyz}.

        residue_scope:
            "auto" – protein residues automatically extracted from PLIP tables
            list   – manual list such as ['SER:B:190', 'VAL:B:213']
            "all"  – use every protein residue (except the ligand)
        """
        if residue_scope != "all":
            keep_set = (
                set(self.hot_residue_ids)
                if residue_scope == "auto"
                else set(residue_scope)
            )
        else:
            keep_set = {
                f"{rec[1]}:{rec[2]}:{rec[3]}"
                for rec in self.atoms
                if not self._is_ligand(rec)
            }

        complex_xyz = self._atoms_to_xyz(self.atoms)
        fragA_atoms: list = []
        fragB_atoms: list = []

        for rec in self.atoms:
            rec_id = f"{rec[1]}:{rec[2]}:{rec[3]}"
            if self._is_ligand(rec):
                fragA_atoms.append(rec)          # ligand real
                fragB_atoms.append(self._ghost(rec))  # ligand ghost in B*
            elif rec_id in keep_set:
                fragA_atoms.append(self._ghost(rec))  # protein ghost in A*
                fragB_atoms.append(rec)          # protein real

        fragA_xyz = self._atoms_to_xyz(fragA_atoms)
        fragB_xyz = self._atoms_to_xyz(fragB_atoms)

        self._geometries = {
            "complex": complex_xyz,
            "fragA": fragA_xyz,
            "fragB": fragB_xyz,
        }
        return self._geometries

    def write_xyz(self, out_dir: str = "./geom"):
        """Write the three XYZ files to disk."""
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for tag, xyz in self._geometries.items():
            (Path(out_dir) / f"{tag}.xyz").write_text(xyz)

    def to_pyscf(self, basis: str = "def2-SVP"):
        """Return a dict of PySCF Mole objects; call build_geometries() first."""
        from pyscf import gto

        moles = {}
        for tag, xyz in self._geometries.items():
            # PySCF accepts element names like 'GhC' for ghost carbon.
            moles[tag] = gto.M(atom=xyz, basis=basis, charge=0, spin=0)
        return moles

    # ---------- internal helpers ----------
    def _parse_pdb(self):
        records = []
        with self.pdb_path.open() as fh:
            for line in fh:
                if not self._pdb_atom_line.match(line):
                    continue
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21].strip()
                resnum = int(line[22:26])
                x, y, z = map(float, (line[30:38], line[38:46], line[46:54]))
                elem = (line[76:78].strip() or name[0]).capitalize()
                records.append((name, resname, chain, resnum, elem, x, y, z))
        return records

    def _parse_plip(self) -> List[str]:
        """
        Extract protein residue identifiers from PLIP tables.
        Looks for `| RESNR | RESTYPE | RESCHAIN |` columns.
        """
        ids: list[str] = []
        pattern = re.compile(r"^\|\s+(\d+)\s+\|\s+(\w{3})\s+\|\s+([A-Z])\s+\|")
        with self.plip_path.open() as fh:
            for line in fh:
                match = pattern.match(line)
                if match:
                    resnr, restype, chain = match.groups()
                    ids.append(f"{restype}:{chain}:{resnr}")
        return ids

    def _is_ligand(self, rec):
        _, resname, chain, *_ = rec
        return chain == self.lig_chain and resname == self.lig_resname

    @staticmethod
    def _ghost(rec):
        """Return a copy of the atom record with element changed to ghost."""
        name, resname, chain, resnum, elem, x, y, z = rec
        return (name, resname, chain, resnum, f"Gh{elem}", x, y, z)

    @staticmethod
    def _atoms_to_xyz(records):
        """Format a list of atom tuples to an XYZ string."""
        lines = [str(len(records)), "generated by CounterpoiseBuilder"]
        for *_, elem, x, y, z in records:
            lines.append(f"{elem:<4s} {x:14.8f} {y:14.8f} {z:14.8f}")
        return "\n".join(lines) + "\n"
