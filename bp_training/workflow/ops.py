import os
from dotenv import load_dotenv
from dagster import op, asset, Output
from pytorch_lightning import Trainer
from mlflow.pytorch import log_model
from bp_training.data import NERData
from bp_training.util.model_io import get_latest_artifact_path
from bp_training.transformer_trainers import TokenClassificationTrainer
from bp_training.trainer import configure_trainer, TrainingConfig

load_dotenv()


# 1. Define an op to initialize and load data using your LightningDataModule
@op
def load_data_op():
    data_module = NERData()
    return data_module


@op
def get_model_from_checkpoint():
    checkpoint_path = get_latest_artifact_path(
        os.environ["MLFLOW_ARTIFACTS_DESTINATION"]
    )
    lightning_module = TokenClassificationTrainer.load_from_checkpoint(checkpoint_path)
    return lightning_module


@op
def get_model_from_hf_hub(data_module):
    lightning_module = TokenClassificationTrainer(
        model_name_or_path="google-bert/bert-base-cased",
        label_list=data_module.label_list,
        eval_splits=["validation", "test"],
        task_name="ner",
    )
    return lightning_module


@op
def get_model(resume, data_module):
    if resume:
        yield get_model_from_checkpoint()
    else:
        yield get_model_from_hf_hub(data_module)


# 2. Define an op for the training step, taking in the data module and model
@op
def train_model(model, data_module, config: TrainingConfig):
    trainer = configure_trainer(config)
    trainer.fit(model, data_module)
    return model
