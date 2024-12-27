from open_biomed.models.foundation_models.molt5 import MolT5

MODEL_REGISTRY = {
    "text_based_molecule_editing": {
        "molt5": MolT5
    },
    "molecule_captioning": {
        "molt5": MolT5
    },
    "text_guided_molecule_generation": {
        "molt5": MolT5
    }
}