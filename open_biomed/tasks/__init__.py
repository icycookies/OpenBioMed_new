from typing import Optional

from open_biomed.tasks.multi_modal_tasks.text_based_molecule_editing import TextMoleculeEditing

TASK_REGISTRY = {
    "text_based_molecule_editing": TextMoleculeEditing
}

def check_compatible(task_name: str, dataset_name: Optional[str], model_name: Optional[str]) -> None:
    # Check if the dataset and model supports the task
    # If not, raise NotImplementedError
    pass