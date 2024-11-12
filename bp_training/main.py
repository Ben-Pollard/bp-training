"""
Entrypoint - currently set up to run a named entity recognition
data module through a token classification trainer
"""

import os
from typing import Optional
import mlflow
import mlflow.runs
from mlflow.entities.experiment import Experiment
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import MLFlowLogger
from dotenv import load_dotenv


from bp_training.data import NERData
from bp_training.transformer_trainers import TokenClassificationTrainer
from bp_training.model_factory import TokenClassificationModel
from bp_training.util.tracking import MLFlowNoSaveModelCheckpoint

if __name__ == "__main__":

    load_dotenv()
    seed_everything(42)

    EXPERIMENT_NAME = "NER Test"
    RUN_NAME = "finetune_test"
    MLFLOW_TRACKING_URI = "http://localhost:5000"
    MLFLOW_ARTIFACTS_DESTINATION = "mlartifacts"

    mlflow.set_experiment(EXPERIMENT_NAME)
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    def get_latest_artifact():
        experiment: Optional[Experiment] = mlflow.get_experiment_by_name("NER Test")
        runs = mlflow.search_runs([experiment.experiment_id])
        latest_run = runs.sort_values("end_time").iloc[-1]
        artifact_path = os.path.join(
            MLFLOW_ARTIFACTS_DESTINATION,
            latest_run["artifact_uri"].split(":/")[-1],
            latest_run["tags.mlflow.latest_checkpoint_artifact"],
        )
        return artifact_path

    data_module = NERData()

    resume = True

    if resume:
        checkpoint = get_latest_artifact()
        lightning_module = TokenClassificationTrainer.load_from_checkpoint(checkpoint)
    else:

        model = TokenClassificationModel(
            model_name_or_path="google-bert/bert-base-cased",
            num_labels=len(data_module.label_list),
        )

        lightning_module = TokenClassificationTrainer(
            model=model(),
            label_list=data_module.label_list,
            eval_splits=["validation", "test"],
            task_name="ner",
        )

    # Enable automatic logging of metrics, parameters, and models
    mlflow.pytorch.autolog()

    with mlflow.start_run(log_system_metrics=True, run_name=RUN_NAME) as run:
        mlflow_logger = MLFlowLogger(
            experiment_name="NER",
            run_id=run.info.run_id,
            run_name=RUN_NAME,
            # tracking_uri=MLFLOW_TRACKING_URI,
            artifact_location="mlartifacts",
            log_model=True,
            synchronous=True,
        )

        checkpoint_callback = MLFlowNoSaveModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )

        trainer = Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=2,
            accelerator="auto",
            devices=1,
            log_every_n_steps=1,
            logger=mlflow_logger,
        )

        trainer.fit(lightning_module, datamodule=data_module)
