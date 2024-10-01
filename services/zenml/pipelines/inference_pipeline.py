from typing import Annotated
import torch
from steps.preprocessing import (
    preprocess_data,
    tensor_converter,
)
from zenml import ArtifactConfig, pipeline
from zenml.client import Client
from zenml.integrations import PytorchIntegration

#zenml experiment-tracker register mlflow_tracker --flavor mlflow
@pipeline(enable_cache=False)
def inference_pipeline(
    data_name: str = "inference_data",
) -> Annotated[
    torch.Tensor,
    ArtifactConfig(name="features", tags=["inference"], is_deployment_artifact=True),
]:
    PytorchIntegration.activate()
    # PandasIntegration.activate()

    client = Client()

    one_hot_encoder = client.get_artifact_version("one_hot_encoder")
    scaler = client.get_artifact_version("scaler")
    categorial_cols = client.get_artifact_version("categorial_cols")

    # Using latest data for inference
    data = client.get_artifact_version(data_name)

    processed_data = preprocess_data(data, categorial_cols, one_hot_encoder, scaler)

    tensor_data = tensor_converter(data=processed_data)

    return tensor_data


if __name__ == "__main__":
    inference_pipeline()
