import os
from typing import Optional
import mlflow
from mlflow.entities.experiment import Experiment


def get_latest_artifact_path(mlflow_artifacts_destination: str):
    experiment: Optional[Experiment] = mlflow.get_experiment_by_name("NER Test")
    runs = mlflow.search_runs([experiment.experiment_id])
    latest_run = runs.sort_values("end_time").iloc[-1]
    artifact_path = os.path.join(
        mlflow_artifacts_destination,
        latest_run["artifact_uri"].split(":/")[-1],
        latest_run["tags.mlflow.latest_checkpoint_artifact"],
    )
    return artifact_path
