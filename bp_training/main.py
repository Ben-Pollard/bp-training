"""
Entrypoint - currently set up to run a named entity recognition
data module through a token classification trainer
"""

from weakref import proxy

import mlflow
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

    mlflow.set_experiment(EXPERIMENT_NAME)
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # resume_run = "finetune_test"
    # experiment = mlflow.get_experiment_by_name("NER Test")
    # experiment.artifact_location
    # mlflow.get_registry_uri()

    data_module = NERData()

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
