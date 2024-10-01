from typing import Annotated, Tuple

from steps.preprocessing import (
    create_train_torch_dataloader,
    fit_encoders_and_scalers,
    preprocess_data,
    preprocess_target,
    split_data,
    training_data_loader,
)
from torch.utils.data import DataLoader
from zenml import ArtifactConfig, pipeline
from zenml.client import Client


@pipeline(enable_cache=False)
def training_pipeline(
    train_data_path: str = "../../data/raw/train.csv",
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

    one_hot_encoder, scaler, label_encoder = fit_encoders_and_scalers(
        data, target, categorial_cols
    )
    processed_data = preprocess_data(data, categorial_cols, one_hot_encoder, scaler)

    target_encoded = preprocess_target(target, label_encoder)

    train_idx, val_idx = split_data(processed_data)

    train_loader, val_loader = create_train_torch_dataloader(
        processed_data, target_encoded, train_idx, val_idx
    )

    return train_loader, val_loader


if __name__ == "__main__":
    training_pipeline("../../data/raw/train.csv")
