from dagster import job, Config, In, op
from bp_training.workflow import ops

# from bp_training.workflow import assets
# from bp_training.workflow import definitions


@job
def training_job():
    data_module = ops.load_data_op()
    model = ops.get_model_op(data_module)
    trained_model = ops.train_model_op(model, data_module)
    # log_model_asset(trained_model)
