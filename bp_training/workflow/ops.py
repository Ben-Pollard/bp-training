import os
from dotenv import load_dotenv
from dagster import op
from pytorch_lightning import Trainer
from bp_training.data import NERData
from bp_training.util.model_io import get_latest_artifact_path
from bp_training.transformer_trainers import TokenClassificationTrainer
from bp_training.workflow.config import GetModelConfig, TrainModelConfig
from pytorch_lightning.loggers import MLFlowLogger
from bp_training.util.tracking import MLFlowNoSaveModelCheckpoint

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
    lightning_module = TokenClassificationTrainer.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path=checkpoint_path
    )
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


@op(required_resource_keys={"mlflow"})
def train_model_op(context, config: TrainModelConfig, model, data_module):

    mlflow = context.resources.mlflow
    mlflow.enable_system_metrics_logging()
    run_info = mlflow.active_run().info

    mlflow.pytorch.autolog()

    # with mlflow.start_run(log_system_metrics=True, run_name=mlflow.run_name) as run:

    mlflow_logger = MLFlowLogger(
        experiment_name=mlflow.experiment_name,
        run_id=run_info.run_id,
        run_name=run_info.run_name,
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

    trainer.fit(model, data_module)
    return trainer.model
