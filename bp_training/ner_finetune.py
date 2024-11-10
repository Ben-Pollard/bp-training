import pytorch_lightning as pl
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
)
import evaluate
from typing import Optional, List
from collections import defaultdict
import torch
import numpy as np


class NERTrainer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
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
        super(NERTrainer, self).__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, config=self.config
        )
        self.metric = evaluate.load("seqeval")
        self.label_list = label_list
        self.val_outs = defaultdict(list)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss = outputs.loss
        outs = {
            "val_loss": val_loss,
            "predictions": outputs.logits,
            "labels": batch.labels,
        }
        for k in outs.keys():
            self.val_outs[k].append(outs[k])

        return outs

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_outs["predictions"]).detach().cpu().numpy()
        labels = torch.cat(self.val_outs["labels"]).detach().cpu().numpy()
        loss = torch.stack(self.val_outs["val_loss"]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(
            self.compute_metrics(predictions=preds, labels=labels), prog_bar=True
        )
        self.val_outs.clear()

    def compute_metrics(self, predictions, labels):
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
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def configure_optimizers(self):
        optimiser = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        return optimiser


pl.seed_everything(42)

from bp_training.data import NERData

dm = NERData()

model = NERTrainer(
    model_name_or_path="google-bert/bert-base-cased",
    num_labels=len(dm.label_list),
    label_list=dm.label_list,
    eval_splits=["validation", "test"],
    task_name="ner",
)

trainer = pl.Trainer(max_epochs=1, accelerator="auto", devices=1, log_every_n_steps=1)
trainer.fit(model, datamodule=dm)
