from typing import Tuple, Annotated
import pandas as pd
import numpy as np
from zenml import step, ArtifactConfig


@step(enable_cache=False)
def training_data_loader(train_data_path: str, target_name: str) -> Tuple[
    Annotated[pd.DataFrame, ArtifactConfig(name="features", tags=["data_preparation"])],
    Annotated[pd.DataFrame, ArtifactConfig(name="target", tags=["data_preparation"])],
]:
    df_train = pd.read_csv(train_data_path, index_col=0)
    target = df_train[target_name].to_frame()
    data = df_train.drop(target_name, axis=1)
    return data, target


@step(enable_cache=False)
def test_data_loader(
    test_data_path: str,
) -> Annotated[
    pd.DataFrame, ArtifactConfig(name="features", tags=["data_preparation"])
]:
    df_test = pd.read_csv(test_data_path, index_col=0)
    return df_test


@step(enable_cache=False)
def handle_missing_values(data: pd.DataFrame, method: str = "drop") -> pd.DataFrame:
    if method == "drop":
        return data.dropna()
    elif method == "mean":
        return data.fillna(data.mean())
    elif method == "median":
        return data.fillna(data.median())
    elif method == "mode":
        return data.fillna(data.mode().iloc[0])
    else:
        raise ValueError(f"Unsupported method: {method}")


@step(enable_cache=False)
def split_data(
    processed_train_data: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> Tuple[
    Annotated[
        np.ndarray, ArtifactConfig(name="train_indices", tags=["data_preparation"])
    ],
    Annotated[
        np.ndarray, ArtifactConfig(name="validation_indices", tags=["data_preparation"])
    ],
]:
    np.random.seed(seed)
    indices = np.random.permutation(len(processed_train_data))
    test_split = int(len(processed_train_data) * test_size)
    train_indices, val_indices = indices[test_split:], indices[:test_split]
    return train_indices, val_indices


@step(enable_cache=False)
def save_train_val_data(
    data: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_data: pd.DataFrame,
    train_file: str = "train_data.csv",
    val_file: str = "val_data.csv",
    test_file: str = "test_data.csv",
):
    train_data = data.iloc[train_indices]
    val_data = data.iloc[val_indices]

    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)
    test_data.to_csv(test_file, index=False)
