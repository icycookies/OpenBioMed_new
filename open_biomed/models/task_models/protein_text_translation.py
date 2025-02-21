from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch

from open_biomed.data import Protein, Text
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.collator import EnsembleCollator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurized, EnsembleFeaturizer

class TextBasedProteinGenerationModel(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        self.supported_tasks["text_based_protein_generation"] = {
            "forward_fn": self.forward_text_based_protein_generation,
            "predict_fn": self.predict_text_based_protein_generation,
            "featurizer": EnsembleFeaturizer({
                "text": self.featurizers["text"],
                "label": self.featurizers["protein"],
            }),
            "collator": EnsembleCollator({
                "text": self.collators["text"],
                "label": self.collators["protein"]
            })
        }

    @abstractmethod
    def forward_text_based_protein_generation(self,
        text: Featurized[Text],
        label: Featurized[Protein],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_text_based_protein_generation(self,
        text: Featurized[Text]
    ) -> List[Protein]:
        raise NotImplementedError