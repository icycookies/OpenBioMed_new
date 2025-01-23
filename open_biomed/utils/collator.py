from abc import ABC, abstractmethod
from typing import Any, Dict, List

from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer, DataCollatorWithPadding

class Collator(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, inputs: List[Any]) -> Any:
        raise NotImplementedError

class PygCollator(Collator):
    def __init__(self, follow_batch: List[str]=[], exclude_keys: List[str]=[]) -> None:
        super().__init__()
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, inputs: List[Data]) -> Batch:
        return Batch.from_data_list(inputs, follow_batch=self.follow_batch, exclude_keys=self.exclude_keys)

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