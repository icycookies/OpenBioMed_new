from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

from open_biomed.data import Protein
from open_biomed.models.functional_model import ProteinDecoder
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import ProteinFeaturizer, ProteinTransformersFeaturizer

class ProGen(ProteinDecoder):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)
        self.model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name_or_path)
        self.featurizer = ProteinTransformersFeaturizer(model_cfg.model_name_or_path)
        self.collator = DataCollatorWithPadding(self.tokenizer)

    def get_protein_featurizer(self) -> ProteinFeaturizer:
        return self.featurizer

    def generate_protein(self, inputs_embeds: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, **kwargs) -> List[Protein]:
        if inputs_embeds is not None:
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **self.config.generation.to_dict(),
            )
        else:
            
            outputs = self.model.generate()