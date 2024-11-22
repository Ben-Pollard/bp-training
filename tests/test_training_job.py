from dagster import DagsterInstance, RunConfig
import dagster_mlflow.hooks
import dagster_mlflow.resources
from bp_training.workflow.jobs import training_job
from bp_training.workflow.config import GetModelConfig, TrainModelConfig
import dagster_mlflow
from dotenv import load_dotenv
import os

load_dotenv()


def test_training_job():

    get_model_config = GetModelConfig(resume=False)

    training_config = TrainModelConfig(
        experiment_name="NER Test",
        run_id="test_run_id",
        run_name="test_run",
        artifact_location="mlartifacts",
        max_epochs=2,
    )

    run_config = RunConfig(
        ops={
            "get_model_op": get_model_config,
            "train_model_op": training_config,
        },
        resources={
            "mlflow": {
                "config": {
                    "experiment_name": "dagster",
                    "mlflow_tracking_uri": os.environ["MLFLOW_TRACKING_URI"],
                    # "mlflow_run_id": "dagster_run",
                }
            }
        },
    )

    instance = DagsterInstance.ephemeral()
    result = training_job.execute_in_process(run_config=run_config, instance=instance)
    assert result.success
