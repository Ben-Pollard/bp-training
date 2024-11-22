from dagster import job, Config, In, op
from bp_training.workflow import ops

# from bp_training.workflow import assets
# from bp_training.workflow import definitions


@job
def training_job():
    # data should be materialised. currently using hf cache
    data_module = ops.load_data_op()

    # untrained model should be stored - we're already using the hf cache + mlflow!
    model = ops.get_model_op(data_module)

    # trained model storage should be managed by mlflow
    trained_model = ops.train_model_op(model=model, data_module=data_module)
    # log_model_asset(trained_model)
