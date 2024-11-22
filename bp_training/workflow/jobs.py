from dagster import job
from dagster_mlflow import end_mlflow_on_run_finished, mlflow_tracking
from bp_training.workflow import ops


# pylint: disable=no-value-for-parameter
@end_mlflow_on_run_finished
@job(resource_defs={"mlflow": mlflow_tracking})
def training_job():
    # data should be materialised. currently using hf cache
    data_module = ops.load_data_op()

    # untrained model should be stored - we're already using the hf cache + mlflow!
    model = ops.get_model_op(data_module)

    # trained model storage should be managed by mlflow
    trained_model = ops.train_model_op(model=model, data_module=data_module)
    # log_model_asset(trained_model)
