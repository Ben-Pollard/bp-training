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
