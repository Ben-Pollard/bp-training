from dagster import job, Config, In, op
from bp_training.workflow import ops

# from bp_training.workflow import assets
# from bp_training.workflow import definitions
from bp_training.trainer import TrainingConfig


class JobConfig(Config):
    resume: bool


@op
def provide_resume() -> bool:
    return False

@job
def training_job():
    job_config = JobConfig(resume=False)
    training_config = TrainingConfig(
        experiment_name="NER Test",
        run_id="test_run_id",
        run_name="test_run",
        artifact_location="mlartifacts",
        max_epochs=2,
    )
    data_module = ops.load_data_op()
    resume = provide_resume()
    model = ops.get_model(resume=resume, data_module=data_module)
    trained_model = ops.train_model(model, data_module, training_config)
    # log_model_asset(trained_model)
