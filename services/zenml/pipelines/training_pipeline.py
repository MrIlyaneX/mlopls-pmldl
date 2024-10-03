import os
from typing import Annotated, Tuple

import zenml
from dotenv import load_dotenv
from steps.basic_preprocessing import (
    handle_missing_values,
    save_train_val_data,
    split_data,
    test_data_loader,
    training_data_loader,
)
from zenml import ArtifactConfig, pipeline
from zenml.client import Client


@pipeline
def training_pipeline(train_data_path: str, test_data_path: str, target_name: str):
    # train_data, target = training_data_loader(train_data_path, target_name)
    # **to save target with the data csv
    train_data = test_data_loader(train_data_path)
    test_data = test_data_loader(test_data_path)

    train_data = handle_missing_values(train_data, method="drop")
    test_data = handle_missing_values(test_data, method="drop")

    train_indices, val_indices = split_data(train_data)

    save_train_val_data(
        train_data,
        train_indices,
        val_indices,
        test_data,
        train_file="/opt/data/processed/train_data.csv",
        val_file="/opt/data/processed/val_data.csv",
        test_file="/opt/data/processed/test_data.csv",
    )


training_pipeline(
    train_data_path="/opt/data/raw/train.csv",
    test_data_path="/opt/data/raw/test.csv",
    target_name="Target",
)
