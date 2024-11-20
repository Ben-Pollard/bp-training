from dagster import execute_job
from bp_training.workflow.jobs import training_job

def test_training_job():
    result = execute_job(
        training_job,
        run_config={}
    )
    assert result.success
