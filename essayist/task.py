import os
from typing import Any, Dict, Literal

import pytorch_lightning as pl
import torch
from transformers import (AutoConfig, AutoModelForCausalLM,
                          get_linear_schedule_with_warmup)
from deepspeed.ops.adam import DeepSpeedCPUAdam


class LanguageModeling(pl.LightningModule):
    """Language Modeling

    Attributes:
        model: model
        num_classes: the number of classes
        total_steps: total training steps for lr scheduling
        learning_rate: Max LR
        warmup_rate: warmup step rate
        model_save_dir: path to save model
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        total_steps: int,
        learning_rate: float,
        warmup_rate: float,
        model_save_dir: str,
        optimizer_name: Literal["adam", "deepspeed"]="adam",
    ):
        super().__init__()

        self.model = model

        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.warmup_rate = warmup_rate
        self.model_save_dir = model_save_dir
        self.optimizer_name = optimizer_name

        self.save_hyperparameters(
            {
                "model": None,
                "total_steps": total_steps,
                "learning_rate": learning_rate,
                "warmup_rate": warmup_rate,
                "model_save_dir": model_save_dir,
            }
        )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, float]:
        """Train step function

        Args:
            batch: training batch input/label
        Returns:
            metrics dictionary of this train step
        """
        outputs = self.model(**batch)

        loss = outputs.loss
        ppl = loss.detach().exp()
        metrics = {"loss": loss, "ppl": ppl}
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        return metrics

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, float]:
        """Validation step function

        Args:
            batch: validating batch input/label
        Returns:
            metrics dictionary of this validation step
        """
        outputs = self.model(**batch)

        loss = outputs.loss
        ppl = loss.detach().exp()
        metrics = {"val-loss": loss, "val-ppl": ppl}
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        return metrics

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, float]:
        """Test step function

        Args:
            batch: testing batch input/label
        Returns:
            metrics dictionary of this test step
        """
        outputs = self.model(**batch)

        loss = outputs.loss
        ppl = loss.detach().exp()
        metrics = {"test-loss": loss, "test-ppl": ppl}
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        return metrics

    def configure_optimizers(self) -> Dict:
        if self.optimizer_name == "deepspeed":
            optimizer = DeepSpeedCPUAdam(
                self.model.parameters(), lr=self.learning_rate
            )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizers = {"optimizer": optimizer}

        if self.warmup_rate is not None:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.total_steps * self.warmup_rate),
                num_training_steps=self.total_steps,
            )
            optimizers["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
                "name": "Learning Rate",
            }

        return optimizers

    def validation_epoch_end(self, outputs) -> None:
        outputs = self.all_gather(outputs)

        if self.trainer.is_global_zero and self.model_save_dir:
            val_losses = [output["val-loss"].mean() for output in outputs]
            val_loss_mean = sum(val_losses) / len(val_losses)

            model_save_path = os.path.join(
                self.model_save_dir,
                f"model-{self.current_epoch:02d}epoch-{self.global_step}steps-{val_loss_mean:.4f}loss",
            )
            torch.save(self.model.state_dict(), model_save_path)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint["model_config"] = self.model.config.to_dict()
        checkpoint["base_model_prefix"] = self.model.base_model_prefix

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        config_dict = checkpoint["model_config"]
        config_cls = AutoConfig.for_model(checkpoint["base_model_prefix"])
        config = config_cls.from_dict(config_dict)
        self.model = AutoModelForCausalLM.from_config(config)
        return super().on_load_checkpoint(checkpoint)
