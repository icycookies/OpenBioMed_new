from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch

from transformers import AutoTokenizer, DataCollatorWithPadding,BatchEncoding
from torch_geometric.data import Batch, Data

from open_biomed.data.molecule import Molecule
from open_biomed.data.protein import Protein
class Collator(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, inputs: List[Any]) -> Any:
        raise NotImplementedError

class EnsembleCollator(Collator):
    def __init__(self, to_ensemble: Dict[str, Collator]) -> None:
        super().__init__()
        self.collators = {}
        for k, v in to_ensemble.items():
            self.collators[k] = v

    def __call__(self, inputs: List[Dict[str, Any]]) -> Dict[Any, Any]:
        collated = {}
        for k in inputs[0]:
            collated[k] = self.collators[k]([item[k] for item in inputs])
        return collated

    def get_attrs(self) -> List[str]:
        return list(self.collators.keys())

class MolCollator(Collator):
    def __call__(self, mol_inputs:List) -> Any:
        return torch.stack([mol.to_torch() for mol in mol_inputs])
class ProCollator(Collator):
    def __call__(self, pro_inputs:List) -> Any:
        return torch.stack([pro.to_torch() for pro in pro_inputs])
class DtiCollator(Collator):
    def __call__(self, inputs:list):
        drug_list,protein_list,label_list=[],[],[]
        for element in inputs:
            drug_list.append(element["drug"])
            protein_list.append(element["protein"])
            label_list.append(element["label"])
        inputs["drug"]=torch.stack([mol.to_torch() for mol in inputs["drug"]])
        inputs["protein"]=torch.stack([pro.to_torch() for pro in inputs["protein"]])
        inputs["label"]=torch.tensor(inputs["label"])
        return inputs
#为了dti这个任务额外添加的collator！

class BaseCollator(ABC):
    def __init__(self, config):
        self.config = config
        self._build(config)

    @abstractmethod
    def __call__(self, data, **kwargs):
        raise NotImplementedError

    def _collate_single(self, data, config):
        if isinstance(data[0], Data):
            return Batch.from_data_list(data)
        elif torch.is_tensor(data[0]):
            return torch.stack([x.squeeze() for x in data])
        elif isinstance(data[0], BatchEncoding):
            return config["collator"](data)
        elif isinstance(data[0], dict):
            result = {}
            for key in data[0]:
                result[key] = self._collate_single([x[key] for x in data], config[key] if key in config else {})
            return result
        elif isinstance(data[0], int):
            return torch.tensor(data).view((-1, 1))

    def _collate_multiple(self, data, config):
        cor = []
        flatten_data = []
        for x in data:
            cor.append(len(flatten_data))
            flatten_data += x
        cor.append(len(flatten_data))
        return (cor, self._collate_single(flatten_data, config),)

    def _build(self, config):
        if not isinstance(config, dict):
            return
        if "model_name_or_path" in config:
            tokenizer = name2tokenizer[config["transformer_type"]].from_pretrained(config["model_name_or_path"])
            if config["transformer_type"] == "gpt2":
                tokenizer.pad_token = tokenizer.eos_token
            config["collator"] = DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding=True
            )
            return
        for key in config:
            self._build(config[key])

class MolCollator(BaseCollator):
    def __init__(self, config):
        super(MolCollator, self).__init__(config)

    def __call__(self, mols):
        if len(self.config["modality"]) > 1:
            batch = {}
            for modality in self.config["modality"]:
                batch[modality] = self._collate_single([mol[modality] for mol in mols], self.config["featurizer"][modality])
        else:
            batch = self._collate_single(mols, self.config["featurizer"]["structure"])
        return batch

class ProteinCollator(BaseCollator):
    def __init__(self, config):
        super(ProteinCollator, self).__init__(config)

    def __call__(self, proteins):
        if len(self.config["modality"]) > 1:
            batch = {}
            for modality in self.config["modality"]:
                if isinstance(proteins[0][modality], list):
                    batch[modality] = self._collate_multiple([protein[modality] for protein in proteins], self.config["featurizer"][modality])
                else:
                    batch[modality] = self._collate_single([protein[modality] for protein in proteins], self.config["featurizer"][modality])
        else:
            batch = self._collate_single(proteins, self.config["featurizer"]["structure"])
        return batch

class TaskCollator(ABC):
    def __init__(self, config):
        super(TaskCollator, self).__init__()
        self.config = config

    @abstractmethod
    def __call__(self, data, **kwargs):
        raise NotImplementedError

class DTICollator(TaskCollator):
    def __init__(self, config):
        super(DTICollator, self).__init__(config)
        self.mol_collator = MolCollator(config["mol"])
        self.protein_collator = ProteinCollator(config["protein"])

    def __call__(self, data):
        mols, prots, labels = map(list, zip(*data))
        return self.mol_collator(mols), self.protein_collator(prots), torch.tensor(labels)