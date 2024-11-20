from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from bp_training.util.tracking import MLFlowNoSaveModelCheckpoint
from dagster import Config


class LoadDataConfig(Config):
    # Define any configuration parameters needed for loading data
    pass


class GetModelConfig(Config):
    resume: bool


class TrainModelConfig(Config):
    experiment_name: str
    run_id: str
    run_name: str
    artifact_location: str
    max_epochs: int


def configure_trainer(config: TrainModelConfig) -> Trainer:
    mlflow_logger = MLFlowLogger(
        experiment_name=config.experiment_name,
        run_id=config.run_id,
        run_name=config.run_name,
        artifact_location=config.artifact_location,
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
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        logger=mlflow_logger,
    )

    return trainer
