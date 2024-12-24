from typing import Any, Dict, List, Tuple
from typing_extensions import Self

import numpy as np

from open_biomed.data.text import Text
from open_biomed.utils.exception import ProteinConstructError

class Protein(dict):
    def __init__(self) -> None:
        super().__init__()
        # basic properties: 1D sequence, 3D coordinates
        self.sequence = None
        self.backbone = None           # Backbone structure
        self.all_atom_coords = None    # All-atom coordinates
        self.chi_angles = None         # Chi-angles of side chains

        # other related properties: textual descriptions and identifier in knowledge graph
        self.description = None
        self.kg_accession = None

    @classmethod
    def from_fasta(cls, fasta: str) -> Self:
        # initialize a protein with a fasta file
        pass

    @classmethod
    def from_fasta_file(cls, fasta_file: str) -> Self:
        # initialize a protein with a fasta file
        pass

    @classmethod
    def from_pdb_file(cls, pdb_file: str) -> Self:
        # initialize a protein with a pdb file
        pass

    @staticmethod
    def folding(sequence: str, method: str='esmfold') -> np.ndarray:
        # Generate the 3D coordinates of the protein backbone
        pass

    @staticmethod
    def inverse_folding(conformer: np.ndarray, method: str='esm-1f') -> str:
        # Generate the amino acid sequence based on protein backbone coordinates
        pass

    @staticmethod
    def convert_chi_angles_to_coordinates(backbone: np.ndarray, chi_angles: np.ndarray) -> np.ndarray:
        # Calculate all-atom coordinates with backbone structure and chi-angles
        pass

    @staticmethod
    def convert_coordinates_to_chi_angles(coordinates: np.ndarray) -> np.ndarray:
        # Calculate chi-angles with all-atom coordinates
        pass

    def _add_sequence(self, base: str='backbone') -> None:
        # Add class property: sequence, based on backbone
        pass

    def _add_backbone(self, method: str='esmfold', base: str='sequence') -> None:
        # Add class property: backbone, based on sequence
        pass
    
    def _add_description(self, text_database: Dict[Any, Text], identifier_key: str='sequence', base: str='sequence') -> None:
        # Add class property: description, based on sequence
        pass

    def _add_kg_accession(self, kg_database: Dict[Any, str], identifier_key: str='sequence', base: str='sequence') -> None:
        # Add class property: kg_accession, based sequence
        pass

    def to_file(self, file: str, format: str='pdb') -> None:
        # Save the protein to a file with a certain format, default: pdb
        pass