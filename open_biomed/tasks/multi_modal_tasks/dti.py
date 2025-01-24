from typing import Dict, List, Tuple, Optional
from typing_extensions import Any

import json
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from open_biomed.utils.config import Config,Struct
from open_biomed.models.base_model import BaseModel
from open_biomed.tasks.base_task import BaseTask, DefaultDataModule, DefaultModelWrapper
from open_biomed.utils.collator import Collator,DTICollator
from open_biomed.utils.featurizer import Featurizer
from open_biomed.utils.dti_utils.metrics import metrics_average

class DrugTargetInteraction(BaseTask):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def get_datamodule(dataset_cfg: Config, featurizer: Featurizer, collator: Collator) -> pl.LightningDataModule:
        return DefaultDataModule("dti",dataset_cfg,featurizer,collator)

    @staticmethod
    def get_model_wrapper(model_cfg: Config, train_cfg: Config)->pl.LightningModule:
        return DefaultModelWrapper("dti",model_cfg,train_cfg)
    @staticmethod
    def get_callbacks(callback_cfg: Optional[Config]=None) -> pl.Callback:
       return DTIEvaluationCallback()
    
    @staticmethod
    def get_monitor_cfg()-> Struct:
        return Struct(
            name="name",
            output_str="path",
            mode="max"
        )
    
class DTIEvaluationCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.outputs = []
        self.eval_dataset = None
       # self.best_metrics = {}

    # def on_validation_batch_end(self, 
    #     trainer: "pl.Trainer", 
    #     pl_module: "pl.LightningModule",
    #     outputs:Optional[STEP_OUTPUT],
    #     batch:Any,
    #     batch_idx: int,
    #     dataloader_idx: int = 0
    #     ) -> None:
    #     super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
    #     self.outputs.extend(outputs)
    #     if batch_idx == 0:
    #         for i in range(2):
    #             out_labels = self.eval_dataset.labels[i]
    #             logging.info(f"Drug: {self.eval_dataset.smiles[i] }")
    #             logging.info(f"Protein: {self.eval_dataset.proteins[i]}")
    #             logging.info(f"Predict: {self.outputs[i]}")
    #             logging.info(f"Ground Truth: {out_labels}")
    
    # def on_validation_epoch_start(self,trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #    super().on_validation_epoch_start(trainer, pl_module)
    #    self.status="val"
    #    self.outputs=[]
    #    self.eval_dataset = trainer.val_dataloader.dataset
    
    # def on_validation_epoch_end(self,trainer: "pl.Trainer",
    # pl_module: "pl.LightningModule") -> None:
    #     super().on_validation_epoch_end(trainer, pl_module)
    #     self.status = "val"

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
