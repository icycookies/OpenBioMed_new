from typing import Optional, Union
from typing_extensions import Any

from absl import logging
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

class Queue:
    def __init__(self, max_len=50):
        self.items = [1]
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)   

class GradientClip(pl.Callback):
    def __init__(self, max_grad_norm: Union[float, str]='Q', Q=Queue(3000)) -> None:
        super().__init__()
        # self.max_norm = max_norm
        self.gradnorm_queue = Q
        if max_grad_norm == 'Q':
            self.max_grad_norm = max_grad_norm
        else:
            self.max_grad_norm = float(max_grad_norm)

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer: torch.optim.Optimizer) -> None:
        # zero graidents if they are not finite
        if self.max_grad_norm == 'Q':
            max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 2 * self.gradnorm_queue.std()
            max_grad_norm = max_grad_norm.item()
        else:
            max_grad_norm = self.max_grad_norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            pl_module.parameters(), max_norm=max_grad_norm, norm_type=2.0
        )

        if self.max_grad_norm == 'Q':
            if float(grad_norm) > max_grad_norm:
                self.gradnorm_queue.add(float(max_grad_norm))
            else:
                self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            logging.info(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {max_grad_norm:.1f}",
            )
        pl_module.log_dict(
            {
                "grad_norm": grad_norm.item(),
                'max_grad_norm': max_grad_norm,
            },
            on_step=True,
            prog_bar=False,
            logger=True,
            batch_size=pl_module.train_cfg.batch_size,
        )

class RecoverCallback(pl.Callback):
    def __init__(self, latest_ckpt: str, recover_trigger_loss: float=1e3, resume: bool=False) -> None:
        super().__init__()
        self.latest_ckpt = latest_ckpt
        self.recover_trigger_loss = recover_trigger_loss
        self.resume = resume

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        if os.path.exists(self.latest_ckpt) and self.resume:
            print(f"recover from checkpoint: {self.latest_ckpt}")
            checkpoint = torch.load(self.latest_ckpt)
            pl_module.load_state_dict(checkpoint["state_dict"])
            # pl_module.load_from_checkpoint(self.latest_ckpt)
        elif not os.path.exists(self.latest_ckpt) and self.resume:
            print(
                f"checkpoint {self.latest_ckpt} not found, training from scratch"
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if "loss" not in outputs:
            return None

        if outputs["loss"] > self.recover_trigger_loss:
            logging.warning(
                f"loss too large: {outputs}\n recovering from checkpoint: {self.latest_ckpt}"
            )
            if os.path.exists(self.latest_ckpt):
                checkpoint = torch.load(self.latest_ckpt)
                pl_module.load_state_dict(checkpoint["state_dict"])
            else:
                for layer in pl_module.children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
                logging.warning(
                    f"checkpoint {self.latest_ckpt} not found, training from scratch"
                )

        else:
            pass