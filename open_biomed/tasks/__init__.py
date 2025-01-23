from typing import Optional

from open_biomed.tasks.multi_modal_tasks.text_based_molecule_editing import TextMoleculeEditing
from open_biomed.tasks.multi_modal_tasks.molecule_text_translation import MoleculeCaptioning
from open_biomed.tasks.multi_modal_tasks.text_guided_molecule_generation import TextGuidedMoleculeGeneration
from open_biomed.tasks.multi_modal_tasks.molecule_question_answering import MoleculeQA
from open_biomed.tasks.multi_modal_tasks.protein_question_answering import ProteinQA
from open_biomed.tasks.aidd_tasks.molecule_property_prediction import MoleculePropertyPrediction

TASK_REGISTRY = {
    "text_based_molecule_editing": TextMoleculeEditing,
    "molecule_captioning": MoleculeCaptioning,
    "text_guided_molecule_generation": TextGuidedMoleculeGeneration,
    "molecule_question_answering": MoleculeQA,
    "protein_question_answering": ProteinQA,
    "molecule_property_prediction": MoleculePropertyPrediction
}

def check_compatible(task_name: str, dataset_name: Optional[str], model_name: Optional[str]) -> None:
    # Check if the dataset and model supports the task
    # If not, raise NotImplementedError
    pass