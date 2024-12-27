from open_biomed.datasets.text_based_molecule_editing_dataset import FSMolEdit
from open_biomed.datasets.molecule_captioning_dataset import CheBI_20

DATASET_REGISTRY = {
    "fs_mol_edit": FSMolEdit,
    "CheBI_20": CheBI_20
}