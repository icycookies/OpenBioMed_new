from typing import Tuple, Union, Any, Dict, Optional, List
from typing_extensions import Self

import json
import os
from torch.utils.data import Dataset

from open_biomed.data import Molecule, Text
from open_biomed.datasets.base_dataset import BaseDataset, assign_split, featurize
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized
import csv

class MoleculeQADataset(BaseDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        self.texts, self.labels = [], []
        super(MoleculeQADataset, self).__init__(cfg, featurizer)

    def __len__(self) -> int:
        return len(self.texts)

    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        return {
            "text": self.texts[index], 
            "label": self.labels[index],
        }

class MoleculeQAEvalDataset(Dataset):
    def __init__(self) -> None:
        super(MoleculeQAEvalDataset, self).__init__()
        self.texts, self.labels = [], []

    @classmethod
    def from_train_set(cls, dataset: MoleculeQADataset) -> Self:
        mol2label = dict()
        for i in range(len(dataset)):
            text = dataset.texts[i]
            label = dataset.labels[i]
            dict_key = str(text)
            if dict_key not in mol2label:
                mol2label[dict_key] = []
            mol2label[dict_key].append((text, label))
        new_dataset = cls()
        for k, v in mol2label.items():
            new_dataset.texts.append(v[0][0])
            new_dataset.labels.append(v[0][1])
        new_dataset.featurizer = dataset.featurizer
        return new_dataset

    def __len__(self) -> int:
        return len(self.texts)

    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        return {
            "text": self.texts[index], 
            "label": self.labels[index],
        }
    
    def get_labels(self) -> List[List[Text]]:
        return self.labels

class MolQA(MoleculeQADataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(MolQA, self).__init__(cfg, featurizer)

    def _load_data(self) -> None:
        self.split_indexes = {}
        cnt = 0
        for split in ["train", "val", "test"]:
            self.split_indexes[split] = []
            with open(os.path.join(self.cfg.path, f"{split}.json"), "r") as f:
                cur = 0
                sample_list = json.loads(f.readlines()[0])
                for sample in sample_list:
                    # Getting the length of a list could be slow
                    self.split_indexes[split].append(cnt)
                    cnt += 1
                    cur += 1

                    self.texts.append(Text.from_str(sample["question"]))
                    self.labels.append(Text.from_str(sample["answer"]))
                    if cur >= 5 and self.cfg.debug:
                        break
        
    @assign_split
    def split(self, split_cfg: Optional[Config] = None) -> Tuple[Any, Any, Any]:
        attrs = ["texts", "labels"]
        ret = (
            self.get_subset(self.split_indexes["train"], attrs), 
            MoleculeQAEvalDataset.from_train_set(self.get_subset(self.split_indexes["val"], attrs)),
            MoleculeQAEvalDataset.from_train_set(self.get_subset(self.split_indexes["test"], attrs)),
        )
        del self
        return ret
