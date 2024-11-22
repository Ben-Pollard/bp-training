import os
from dotenv import load_dotenv
from dagster import op
from bp_training.data import NERData
from bp_training.util.model_io import get_latest_artifact_path
from bp_training.transformer_trainers import TokenClassificationTrainer
from bp_training.workflow.config import configure_trainer
from bp_training.workflow.config import LoadDataConfig, GetModelConfig, TrainModelConfig

load_dotenv()


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
def get_model_op(config: GetModelConfig, data_module):
    resume = config.resume
    if resume:
        return get_model_from_checkpoint()
    else:
        return get_model_from_hf_hub(data_module)


@op
def train_model_op(config: TrainModelConfig, model, data_module):
    trainer = configure_trainer(config)
    trainer.fit(model, data_module)
    return trainer.model
