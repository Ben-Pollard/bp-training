from dagster import execute_job, DagsterInstance, RunConfig
from bp_training.workflow.jobs import training_job
from bp_training.workflow.config import TrainingConfig


def test_training_job():
    job_config = {"get_model": {"resume": False}}

    training_config = {
        "train_model": {
            "experiment_name": "NER Test",
            "run_id": "test_run_id",
            "run_name": "test_run",
            "artifact_location": "mlartifacts",
            "max_epochs": 2,
        }
    }

    run_config = RunConfig(job_config | training_config)

    instance = DagsterInstance.ephemeral()
    result = training_job.execute_in_process(run_config=config, instance=instance)
    assert result.success
