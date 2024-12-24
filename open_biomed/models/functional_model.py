from abc import ABC, abstractmethod
from typing import Any, List, Union

import torch

from open_biomed.data import Molecule, Text
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import MoleculeFeaturizer, TextFeaturizer

class MoleculeModel(BaseModel):
    def __init__(self, model_cfg: Config) -> None:
        super(MoleculeModel, self).__init__(model_cfg)
        self.molecule_featurizer = self.get_molecule_featurizer()

    @abstractmethod
    def get_molecule_featurizer(self) -> MoleculeFeaturizer:
        raise NotImplementedError

class MoleculeEncoder(MoleculeModel):
    def __init__(self, model_cfg: Config) -> None:
        super(MoleculeEncoder, self).__init__(model_cfg)

    @abstractmethod
    def encode_molecule(self, molecule: Union[List[Molecule], Any]) -> torch.Tensor:
        raise NotImplementedError

class MoleculeDecoder(MoleculeModel):
    def __init__(self, model_cfg: Config) -> None:
        super(MoleculeDecoder, self).__init__(model_cfg)

class TextEncoder(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super(TextEncoder, self).__init__(model_cfg)
        self.text_featurizer = self.get_text_featurizer()

    @abstractmethod
    def get_text_featurizer(self) -> TextFeaturizer:
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: Union[List[Text], Any]) -> torch.Tensor:
        raise NotImplementedError