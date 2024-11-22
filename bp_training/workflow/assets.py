# 3. Define an asset to log and track the trained model as an artifact
@asset
def log_model_asset(trained_model):
    log_model(trained_model, "trained_model")
    return trained_model
