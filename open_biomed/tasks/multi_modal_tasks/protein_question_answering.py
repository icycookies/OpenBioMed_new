from typing import Dict, List, Tuple, Optional
from typing_extensions import Any

import json
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from open_biomed.data import molecule_fingerprint_similarity, check_identical_molecules
from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config, Struct
from open_biomed.utils.featurizer import Featurizer

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import BertTokenizerFast
import numpy as np

class ProteinQA(BaseTask):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("protein_question_answering", dataset_cfg, featurizer, collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config) -> pl.LightningModule:
        return DefaultModelWrapper("protein_question_answering", model_cfg, train_cfg)

    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
        return ProteinQAEvaluationCallback()

    @staticmethod
    def get_monitor_cfg() -> Struct:
        return Struct(
            name="val/ROUGE-1",
            output_str="-BLUE-2_{val/BLUE-2:.4f}-BLUE-4_{val/BLUE-4:.4f}-BLUE-L_{val/BLUE-L:.4f}-ROUGE-1_{val/ROUGE-1:.4f}-ROUGE-2_{val/ROUGE-2:.4f}",
            mode="max",
        )

class ProteinQAEvaluationCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = []
        self.eval_dataset = None

    def on_validation_batch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Optional[STEP_OUTPUT], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        self.outputs.extend(outputs)
        if batch_idx == 0:
            for i in range(2):
                out_labels = str(self.eval_dataset.labels[i])
                logging.info(f"Original: {self.eval_dataset.texts[i]}")
                logging.info(f"Predict: {self.outputs[i]}")
                logging.info(f"Ground Truth: {out_labels}")

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_validation_epoch_start(trainer, pl_module)
        self.status = "val"
        self.outputs = []
        self.eval_dataset = trainer.val_dataloaders.dataset

    def on_validation_epoch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

        gts = self.eval_dataset.labels
        output_tokens = []
        gt_tokens = []
        meteor_scores = []
        rouge_scores_1 = []
        rouge_scores_2 = []
        rouge_scores_L = []

        abs_path = os.path.abspath("./open_biomed/tokenizers/bert-base-uncased/")
        tokenizer = BertTokenizerFast.from_pretrained(abs_path)

        for i in range(len(self.outputs)):
            rouge = scorer.score(str(self.outputs[i]), str(gts[i]))
            rouge_1 = rouge['rouge1'].fmeasure
            rouge_scores_1.append(rouge_1)
            rouge_2 = rouge['rouge2'].fmeasure
            rouge_scores_2.append(rouge_2)
            rouge_L = rouge['rougeL'].fmeasure
            rouge_scores_L.append(rouge_L)

            output_tokens.append(tokenizer.tokenize(str(self.outputs[i]), truncation=True, max_length=512, padding='max_length'))
            output_tokens[i] = list(filter(('[PAD]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[CLS]').__ne__, output_tokens[i]))
            output_tokens[i] = list(filter(('[SEP]').__ne__, output_tokens[i]))

            gt_tokens.append(tokenizer.tokenize(str(gts[i]), truncation=True, max_length=512, padding='max_length'))
            gt_tokens[i] = list(filter(('[PAD]').__ne__, gt_tokens[i]))
            gt_tokens[i] = list(filter(('[CLS]').__ne__, gt_tokens[i]))
            gt_tokens[i] = [list(filter(('[SEP]').__ne__, gt_tokens[i]))]

            # meteor_scores.append(meteor_score(gt_tokens[i], output_tokens[i]))

        blue2 = corpus_bleu(gt_tokens, output_tokens, weights=(0.5, 0.5))
        blue4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))
             
        output_str = "\t".join([str(self.outputs[i]), 
                                    f"{np.mean(blue2):.4f}",  f"{np.mean(blue4):.4f}",
                                    f"{np.mean(rouge_scores_1):.4f}", f"{np.mean(rouge_scores_2):.4f}",
                                    f"{np.mean(rouge_scores_L):.4f}"]) + "\n"

        output_path = os.path.join(trainer.default_root_dir, f"{self.status}_outputs", f"epoch{pl_module.current_epoch}")
        out_metrics = {
            f"{self.status}/BLEU-2": np.mean(blue2),
            f"{self.status}/BLUE-4": np.mean(blue4),
            # f"{self.status}/Meteor": np.mean(meteor_scores),
            f"{self.status}/ROUGE-1": np.mean(rouge_scores_1),
            f"{self.status}/ROUGE-2": np.mean(rouge_scores_2),
            f"{self.status}/ROUGE-L": np.mean(rouge_scores_L)
        }
        pl_module.log_dict(out_metrics)
        print(json.dumps(out_metrics, indent=4))
        json.dump(out_metrics, open(output_path + "_metrics.json", "w"))
        
        with open(output_path + "_outputs.txt", "w") as f:
            f.write(output_str)
        
        
    def on_test_batch_end(self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_test_epoch_start(trainer, pl_module)
        self.status = "test"
        self.outputs = []
        self.eval_dataset = trainer.test_dataloaders.dataset

    def on_test_epoch_end(self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> None:
        self.on_validation_epoch_end(trainer, pl_module)