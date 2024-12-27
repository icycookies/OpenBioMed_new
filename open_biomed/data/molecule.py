from typing import Any, Dict, List, Tuple
from typing_extensions import Self

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AllChem import RWMol
from torch_geometric.data import Data as Graph

from open_biomed.data.text import Text
from open_biomed.utils.exception import MoleculeConstructError

class Molecule:
    def __init__(self) -> None:
        super().__init__()
        # basic properties: 1D SMILES/SELFIES strings, RDKit mol object, 2D graphs, 3D coordinates
        self.smiles = None
        self.selfies = None
        self.rdmol = None
        self.graph = None
        self.conformer = None

        # other related properties: image, textual descriptions and identifier in knowledge graph
        self.img = None
        self.description = None
        self.kg_accession = None

    @classmethod
    def from_smiles(cls, smiles: str) -> Self:
        # initialize a molecule with a SMILES string
        molecule = cls()
        molecule.smiles = smiles
        # molecule._add_rdmol(base="smiles")
        return molecule

    @classmethod
    def from_rdmol(cls, rdmol: RWMol) -> Self:
        # initialize a molecule with a RDKit molecule
        pass

    @classmethod
    def from_pdb_file(cls, pdb_file: str) -> Self:
        # initialize a molecule with a pdb file
        pass

    @classmethod
    def from_image_file(cls, image_file: str) -> Self:
        # initialize a molecule with a image file
        pass

    @staticmethod
    def convert_smiles_to_rdmol(smiles: str, canonical: bool=True) -> RWMol:
        # Convert the smiles string into rdkit mol
        # If the smiles is invalid, raise MolConstructError
        pass

    @staticmethod
    def convert_rdmol_to_graph(rdmol: RWMol) -> Graph:
        # Convert rdkit mol into a (pyg) graph with atom type [N] and edge index [M, 2] 
        pass

    @staticmethod
    def generate_conformer(rdmol: RWMol, method: str='mmff', num_conformers: int=1) -> np.ndarray:
        # Generate 3D conformer with algorithms in RDKit
        # TODO: identify if ML-based conformer generation can be applied
        pass

    def _add_smiles(self, base: str='rdmol') -> None:
        # Add class property: smiles, based on selfies / rdmol / graph, default: rdmol
        pass

    def _add_selfies(self, base: str='smiles') -> None:
        # Add class property: selfies, based on smiles / selfies / rdmol / graph, default: smiles
        pass

    def _add_rdmol(self, base: str='smiles') -> None:
        # Add class property: rdmol, based on smiles / selfies / graph, default: smiles
        if self.rdmol is not None:
            return
        if base == 'smiles':
            self.rdmol = Chem.MolFromSmiles(self.smiles)

    def _add_graph(self, base: str='rdmol') -> None:
        # Add class property: graph, based on smiles / selfies / rdmol, default: rdmol
        pass

    def _add_conformer(self, method: str='mmff', num_conformers: int=1, base: str='rdmol') -> None:
        # Add class property: conformer, based on smiles / selfies / rdmol, default: rdmol
        pass
    
    def _add_description(self, text_database: Dict[Any, Text], identifier_key: str='SMILES', base: str='smiles') -> None:
        # Add class property: description, based on smiles / selfies / rdmol, default: smiles
        pass

    def _add_kg_accession(self, kg_database: Dict[Any, str], identifier_key: str='SMILES', base: str='smiles') -> None:
        # Add class property: kg_accession, based on smiles / selfies / rdmol, default: smiles
        pass

    def to_file(self, file: str, format: str='binary') -> None:
        # Save the molecule to a file with a format, default: binary
        pass

    def __str__(self) -> str:
        return self.smiles

def molecule_fingerprint_similarity(mol1: Molecule, mol2: Molecule, fingerprint_type: str="morgan") -> float:
    # Calculate the fingerprint similarity of two molecules
    try:
        mol1._add_rdmol()
        mol2._add_rdmol()
        if fingerprint_type == "morgan":
            fp1 = AllChem.GetMorganFingerprint(mol1.rdmol, 2)
            fp2 = AllChem.GetMorganFingerprint(mol2.rdmol, 2)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        if fingerprint_type == "rdkit":
            fp1 = Chem.RDKFingerprint(mol1.rdmol)
            fp2 = Chem.RDKFingerprint(mol2.rdmol)
        if fingerprint_type == "maccs":
            fp1 = MACCSkeys.GenMACCSKeys(mol1.rdmol)
            fp2 = MACCSkeys.GenMACCSKeys(mol2.rdmol)
        return DataStructs.FingerprintSimilarity(
            fp1, fp2,
            metric=DataStructs.TanimotoSimilarity
        )
    except Exception:
        return 0.0

def check_identical_molecules(mol1: Molecule, mol2: Molecule) -> bool:
    # Check if the two molecules are the same
    try:
        mol1._add_rdmol()
        mol2._add_rdmol()
        return Chem.MolToInchi(mol1.rdmol) == Chem.MolToInchi(mol2.rdmol)
    except Exception:
        return False