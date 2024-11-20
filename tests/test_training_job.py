from dagster import execute_job
from bp_training.workflow.jobs import training_job

def test_training_job():
    result = execute_job(
        training_job,
        run_config={
            "ops": {
                "train_model": {
                    "config": {
                        "experiment_name": "NER Test",
                        "run_id": "test_run_id",
                        "run_name": "test_run",
                        "artifact_location": "mlartifacts",
                        "max_epochs": 2,
                    }
                }
            }
        }
    )
    assert result.success
