import os
from dotenv import load_dotenv
from dagster import op, Config
from bp_training.data import NERData
from bp_training.util.model_io import get_latest_artifact_path
from bp_training.transformer_trainers import TokenClassificationTrainer
from bp_training.workflow.config import configure_trainer

load_dotenv()


class LoadDataConfig(Config):
    # Define any configuration parameters needed for loading data
    pass


@op(config_schema=LoadDataConfig)
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


class GetModelConfig(Config):
    resume: bool


@op(config_schema=GetModelConfig)
def get_model_op(context, data_module):
    resume = context.op_config["resume"]
    if resume:
        yield get_model_from_checkpoint()
    else:
        yield get_model_from_hf_hub(data_module)


class TrainModelConfig(Config):
    experiment_name: str
    run_id: str
    run_name: str
    artifact_location: str
    max_epochs: int


@op(config_schema=TrainModelConfig)
def train_model_op(context, model, data_module):
    trainer = configure_trainer(context.op_config)
    trainer.fit(model, data_module)
    return model
