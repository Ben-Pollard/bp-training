from dagster import job, Config
from bp_training.workflow import ops

# from bp_training.workflow import assets
# from bp_training.workflow import definitions
from bp_training.trainer import TrainingConfig


class JobConfig(Config):
    resume: bool


@job
def training_job(job_config: JobConfig, training_config: TrainingConfig):
    data_module = ops.load_data_op()
    model = ops.get_model(job_config.resume, data_module)
    trained_model = ops.train_model(model, data_module, training_config)
    # log_model_asset(trained_model)
