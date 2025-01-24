from open_biomed.models.task_models.text_based_molecule_editing import *
from open_biomed.models.task_models.molecule_captioning import *
from open_biomed.models.task_models.text_guided_molecule_generation import *
from open_biomed.models.task_models.dti_model import *
from open_biomed.models.task_models.dti_modelsConfig.molecule.cnn import *
from open_biomed.models.task_models.dti_modelsConfig.protein.cnn import *
SUPPORTED_MOL_ENCODER={
    "cnn": MolCNN,
}

SUPPORTED_PROTEIN_ENCODER={
    "cnn": ProtCNN,
}