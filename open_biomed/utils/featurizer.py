from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from functools import wraps
import torch
from transformers import AutoTokenizer, BatchEncoding
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from open_biomed.data import Molecule, Protein, Text

T = TypeVar('T', bound=Any)

class Featurized(Generic[T]):
    def __init__(self, value: T):
        self.value = value

class Featurizer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_attrs(self) -> List[str]:
        raise NotImplementedError

class MoleculeFeaturizer(Featurizer):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, molecule: Molecule) -> Dict[str, Any]:
        raise NotImplementedError

    def get_attrs(self) -> List[str]:
        return ["molecule"]

class MoleculeTransformersFeaturizer(MoleculeFeaturizer):
    def __init__(self,
        tokenizer: str,
        max_length: int=512,
        add_special_tokens: bool=True,
        base: str='SMILES',
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, model_max_length=max_length, truncation=True)
        self.add_special_tokens = add_special_tokens
        self.base = base
        if base not in "SMILES" and "SELFIES":
            raise ValueError("{base} is not a valid 1D representaiton of molecules!")

    def __call__(self, molecule: Molecule) -> BatchEncoding:
        if self.base == "SMILES":
            molecule._add_smiles()
            parse_str = molecule.smiles
        if self.base == "SELFIES":
            molecule._add_selfies()
            parse_str = molecule.selfies
        return self.tokenizer(
            parse_str, 
            truncation=True, 
            add_special_tokens=self.add_special_tokens,
        )

class MoleculeOneHotFeaturizer(MoleculeFeaturizer):
    smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']
    def __init__(self,config):
        super(MoleculeOneHotFeaturizer, self).__init__()
        self.max_len = config["max_len"]
        self.enc = OneHotEncoder().fit(np.array(self.smiles_char).reshape(-1, 1))
    #分析判断，这里输入进来的数据类型，应该是Molecule类型的
    def __call__(self, molecule: Molecule):
        temp = [c if c in self.smiles_char else '?' for c in molecule.smiles]
      #  print(temp)
        if len(temp) < self.max_len:
            temp = temp + ['?'] * (self.max_len - len(temp))
        else:
            temp = temp[:self.max_len]
       # print(temp)
        return torch.tensor(self.enc.transform(np.array(temp).reshape(-1, 1)).toarray().T)

class TextFeaturizer(Featurizer):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, text: Text) -> List[Any]:
        raise NotImplementedError

    def get_attrs(self) -> List[str]:
        return ["text"]

class TextTransformersFeaturizer(TextFeaturizer):
    def __init__(self, 
        tokenizer: str,
        max_length: int=512,
        add_special_tokens: bool=True,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, model_max_length=max_length, truncation=True)
        self.add_special_tokens = add_special_tokens

    def __call__(self, text: Text) -> BatchEncoding:
        return self.tokenizer(
            text.str, 
            truncation=True, 
            add_special_tokens=self.add_special_tokens,
        )

class EnsembleFeaturizer(Featurizer):
    def __init__(self, to_ensemble: Dict[str, Featurizer]) -> None:
        super().__init__()
        self.featurizers = {}
        for k, v in to_ensemble.items():
            self.featurizers[k] = v

    def __call__(self, **kwargs: Any) -> Dict[Any, Any]:
        featurized = {}
        for k, v in kwargs.items():
            featurized[k] = self.featurizers[k](v)
        return featurized

    def get_attrs(self) -> List[str]:
        return list(self.featurizers.keys())

class ProteinFeaturizer(Featurizer):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, protein: Protein) -> Dict[str, Any]:
        raise NotImplementedError

    def get_attrs(self) -> List[str]:
        return ["protein"]

class ProteinOneHotFeaturizer(ProteinFeaturizer):
    amino_char = [
        '?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L',
        'O', 'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z'
    ]
    
    def __init__(self, config):
        super(ProteinOneHotFeaturizer, self).__init__()
        self.max_length = config["max_length"]
        self.enc = OneHotEncoder().fit(np.array(self.amino_char).reshape(-1, 1))

    def __call__(self, protein:Protein):
        temp = [i if i in self.amino_char else '?' for i in protein.sequence]
       # print("Protein temp: ", temp)
        if len(temp) < self.max_length:
            temp = temp + ['?'] * (self.max_length - len(temp))
        else:
            temp = temp[:self.max_length]
       # print("P T: ", temp)
        return torch.tensor(self.enc.transform(np.array(temp).reshape(-1, 1)).toarray().T)

# class DummyFeaturizer(Featurizer):
#     def __init__(self, attr_name: str):
#         self.attr_name = attr_name

#     def __call__(self, data: Any) -> Dict[str, Any]:
#         return {self.attr_name: data}

#     def get_attrs(self) -> List[str]:
#         return [self.attr_name]
class DummyFeaturizer(Featurizer):
    def __call__(self, data: Any) -> Any:
        return data

    def get_attrs(self) -> List[str]:
        return []