from typing import Annotated, Any, Tuple

import numpy as np
import pandas as pd
import torch
from steps.preprocessing import (
    create_test_torch_dataloader,
    create_train_torch_dataloader,
    fit_encoders_and_scalers,
    preprocess_data,
    preprocess_target,
    split_data,
    tensor_converter,
    test_data_loader,
    training_data_loader,
)
from torch.utils.data import DataLoader
from zenml import ArtifactConfig, pipeline, save_artifact, step
from zenml.client import Client
from zenml.integrations import PytorchIntegration


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


@pipeline(enable_cache=False)
def preprocessing_pipeline(
    data: pd.DataFrame, target: pd.DataFrame, categorial_cols: list
) -> Tuple[
    Annotated[pd.DataFrame, ArtifactConfig(name="features", tags=["data_preparation"])],
    Annotated[np.ndarray, ArtifactConfig(name="target", tags=["data_preparation"])],
]:
    one_hot_encoder, scaler, label_encoder = fit_encoders_and_scalers(
        data, target, categorial_cols
    )
    processed_data = preprocess_data(data, categorial_cols, one_hot_encoder, scaler)

    target_encoded = preprocess_target(target, label_encoder)

    return processed_data, target_encoded


@pipeline(enable_cache=False)
def training_pipeline(
    train_data_path: str = "../data/raw/train.csv",
) -> Tuple[
    Annotated[
        DataLoader, ArtifactConfig(name="train_dataloader", tags=["data_preparation"])
    ],
    Annotated[
        DataLoader,
        ArtifactConfig(name="validation_dataloader", tags=["data_preparation"]),
    ],
]:
    data, target = training_data_loader(train_data_path, target_name="Target")

    categorial_cols = Client().get_artifact_version("categorial_cols")

    processed_data, target_encoded = preprocessing_pipeline(
        data, target, categorial_cols
    )

    train_idx, val_idx = split_data(processed_data)

    train_loader, val_loader = create_train_torch_dataloader(
        processed_data, target_encoded, train_idx, val_idx
    )

    return train_loader, val_loader


training_pipeline("data/raw/train.csv")
