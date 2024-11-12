"""
Pytorch Lightning modules for training transformers
"""

from collections import defaultdict
from typing import List, Optional

import evaluate
import numpy as np
import pytorch_lightning as pl
import torch
from loguru import logger
from transformers import AdamW
from transformers.tokenization_utils_base import BatchEncoding


# pylint: disable=unused-argument arguments-differ
class TokenClassificationTrainer(pl.LightningModule):
    """
    Pytorch Lightning module for training a transformers model on token classification
    """

    def __init__(  # type: ignore
        self,
        model,
        label_list: List[str],
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        """
        Initializes the NERTrainer with the given parameters.

        Args:
            model_name_or_path (str): Path to the pre-trained model or model identifier
              from huggingface.co/models.
            num_labels (int): Number of labels for classification.
            label_list (List[str]): List of label names.
            task_name (str): Name of the task.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 2e-5.
            adam_epsilon (float, optional): Epsilon for the Adam optimizer. Defaults to 1e-8.
            warmup_steps (int, optional): Number of warmup steps for learning rate scheduler.
              Defaults to 0.
            weight_decay (float, optional): Weight decay for the optimizer. Defaults to 0.0.
            train_batch_size (int, optional): Batch size for training. Defaults to 32.
            eval_batch_size (int, optional): Batch size for evaluation. Defaults to 32.
            eval_splits (Optional[list], optional): List of evaluation splits. Defaults to None.
        """
        super().__init__()

        self.save_hyperparameters(ignore="model")

        self.model = model
        self.metric = evaluate.load("seqeval")
        self.label_list = label_list
        self.val_outs = defaultdict(list)
        self.train_outs = defaultdict(list)

    def forward(self, **inputs) -> torch.Tensor:
        """
        Forward pass for the model.

        Args:
            **inputs: Arbitrary keyword arguments for model inputs.

        Returns:
            torch.Tensor: Model outputs.
        """
        return self.model(**inputs)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Performs a single training step.

        Args:
            batch (dict): A batch of training data.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the loss.
        """
        outputs = self(**batch)
        outs = {"loss": outputs.loss}
        for k, v in outs.items():
            self.train_outs[k].append(v)
        return outs

    def validation_step(self, batch: BatchEncoding, batch_idx: int) -> dict:
        """
        Performs a single validation step.

        Args:
            batch (dict): A batch of validation data.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing validation loss, predictions, and labels.
        """
        outputs = self(**batch)
        outs = {
            "val_loss": outputs.loss,
            "predictions": outputs.logits,
            "labels": batch.labels,
        }
        for k, v in outs.items():
            self.val_outs[k].append(v)

        return outs

    def on_train_epoch_end(self) -> None:
        """
        Called at the end of the training epoch to log the average loss.
        """
        loss = torch.stack(self.train_outs["loss"]).mean()
        self.log("loss", loss, prog_bar=True)
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of the validation epoch to log metrics and clear outputs.
        """
        preds = torch.cat(self.val_outs["predictions"]).detach().cpu().numpy()
        labels = torch.cat(self.val_outs["labels"]).detach().cpu().numpy()
        loss = torch.stack(self.val_outs["val_loss"]).mean()
        self.log("val_loss", loss, prog_bar=True)
        metrics = self.compute_metrics(predictions=preds, labels=labels)
        self.val_outs.clear()
        if metrics:
            self.log_dict(metrics, prog_bar=True)
        else:
            logger.warning(f"Metrics undefined for epoch {self.current_epoch}")

    def compute_metrics(
        self, predictions: np.ndarray, labels: np.ndarray
    ) -> Optional[dict]:
        """
        Computes evaluation metrics.

        Args:
            predictions (np.ndarray): Model predictions.
            labels (np.ndarray): True labels.

        Returns:
            dict: Dictionary containing precision, recall, f1, and accuracy.
        """
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(
            predictions=true_predictions, references=true_labels
        )
        if results:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
        else:
            return None

    def configure_optimizers(self) -> AdamW:
        """
        Configures the optimizer for training.

        Returns:
            AdamW: The configured optimizer.
        """
        optimiser = AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            eps=self.hparams["adam_epsilon"],
        )
        return optimiser
