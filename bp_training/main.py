"""
Entrypoint - currently set up to run a named entity recognition
data module through a token classification trainer
"""

import os

import mlflow
import git
from pytorch_lightning import Trainer, seed_everything, callbacks
from pytorch_lightning.loggers import MLFlowLogger
from dotenv import load_dotenv

from bp_training.data import NERData
from bp_training.transformer_trainers import TokenClassificationTrainer

if __name__ == "__main__":

    load_dotenv()
    seed_everything(42)

    mlflow.set_experiment("NER Test")

    data_module = NERData()

    model = TokenClassificationTrainer(
        model_name_or_path="google-bert/bert-base-cased",
        num_labels=len(data_module.label_list),
        label_list=data_module.label_list,
        eval_splits=["validation", "test"],
        task_name="ner",
    )

    # Log the current git hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    mlflow.log_param("git_hash", sha)

    # Enable automatic logging of metrics, parameters, and models
    mlflow.pytorch.autolog()

    # Define checkpoint callback
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    with mlflow.start_run(log_system_metrics=True, run_name="finetune_test") as run:
        mlflow_logger = MLFlowLogger(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            experiment_name="NER",
            log_model=True,
            run_id=run.info.run_id,
        )

        trainer = Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=2,
            accelerator="auto",
            devices=1,
            log_every_n_steps=1,
            logger=mlflow_logger,
        )

        trainer.fit(model, datamodule=data_module)
