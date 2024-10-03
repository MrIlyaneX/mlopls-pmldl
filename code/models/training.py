import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from typing import List
import os
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature

import pandas as pd

from model import BasicNet
from features import load_data, save_encoders, preprocess, fit_encoders
from inference import ModelInferenceInterface

registry_uri = "http://127.0.0.1:8090"
mlflow.set_tracking_uri(uri=registry_uri)


def train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=1,
    device="cpu",
    scheduler=None,
    tolerance=-1,
    tolerance_delta=1e-4,
):
    best = 0.0

    not_improving = 0
    last_loss = None

    for epoch in range(epochs):
        train_loop = tqdm(
            enumerate(train_loader, 0), total=len(train_loader), desc=f"Epoch {epoch}"
        )
        model.train()
        train_loss = 0.0

        for data in train_loop:
            data = data[1]
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            output = model(inputs)

            loss = loss_fn(output, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            train_loop.set_postfix({"loss": loss.item()})

            if scheduler:
                scheduler.step()

        mlflow.log_metric("loss", f"{train_loss / len(train_loader):6f}", step=epoch)

        correct = 0
        total = 0

        with torch.no_grad():
            model.eval()
            val_loop = tqdm(enumerate(val_loader, 0), total=len(val_loader), desc="Val")
            for data in val_loop:
                data = data[1]
                inputs, labels = data[0].to(device), data[1].to(device)

                output = model(inputs)
                loss = loss_fn(output, labels).item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
                total += len(labels)

                val_loop.set_postfix({"acc": correct / total})

            val_acc = correct / total
            mlflow.log_metric(
                "validation loss", f"{loss / len(val_loader):6f}", step=epoch
            )
            mlflow.log_metric("validation accuracy", f"{val_acc:2f}", step=epoch)
            if val_acc > best:
                # torch.save(model.state_dict(), "model_best.pt")
                # torch.save(optimizer.state_dict(), "opimizer.pt")
                best = correct / total
        if epoch != 0:
            if abs(train_loss - last_loss) < tolerance_delta:
                not_improving += 1
                if not_improving == tolerance:
                    print("Stop due to early reaching tolerance_delta")
                    break
            else:
                not_improving = 0
        last_loss = train_loss

    print(best)


def mlflow_training(
    train_file: str = "../../data/processed/train_data.csv",
    val_file: str = "../../data/processed/val_data.csv",
    test_file: str = "../../data/processed/test_data.csv",
    target_name: str = "Target",
    model_name: str = "BasicModel",
    model_alias: str = "Champion",
    registry_uri: str = "127.0.0.1:8000",
    batch_size: int = 32,
    epochs: int = 5,
    categorical_cols: List | None = None,
):
    for file_path in [train_file, val_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    with mlflow.start_run() as run:
        train_data_, val_data_ = load_data(train_file, val_file)

        one_hot_encoder, scaler, label_encoder = fit_encoders(
            train_data_, target_name, categorical_cols
        )
        save_encoders(one_hot_encoder, scaler, label_encoder, categorical_cols)

        # transofrms
        train_target = train_data_[target_name]
        val_target = val_data_[target_name]

        train_data = train_data_.drop(columns=[target_name])
        val_data = val_data_.drop(columns=[target_name])

        train_features = preprocess(
            train_data, one_hot_encoder, scaler, cat_columns=categorical_cols
        )
        train_target_encoded = label_encoder.transform(train_target)

        val_features = preprocess(
            val_data, one_hot_encoder, scaler, cat_columns=categorical_cols
        )

        val_target_encoded = label_encoder.transform(val_target)

        # dataloaders
        train_dataset = TensorDataset(
            torch.tensor(train_features.values, dtype=torch.float32),
            torch.tensor(train_target_encoded, dtype=torch.int64),
        )
        val_dataset = TensorDataset(
            torch.tensor(val_features.values, dtype=torch.float32),
            torch.tensor(val_target_encoded, dtype=torch.int64),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # model init + training
        input_size = train_features.values.shape[1]
        print(input_size)
        model = BasicNet(input_size=input_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_function = nn.CrossEntropyLoss()

        train(
            model,
            optimizer,
            loss_function,
            train_loader,
            val_loader,
            device="cpu",
            epochs=epochs,
        )
        torch.save(model.state_dict(), f"./{model_name}.pt")

        # Mlflow
        model_with_inference = ModelInferenceInterface(
            model,
            cat_columns=categorical_cols,
        )
        model_with_inference.one_hot_encoder = one_hot_encoder
        model_with_inference.scaler = scaler
        model_with_inference.label_encoder = label_encoder
        model_with_inference.model = model
        model_with_inference.cat_columns = categorical_cols

        signature = infer_signature(
            train_data,
            model_with_inference.predict(None, train_data),
        )
        model_info = mlflow.pyfunc.log_model(
            artifact_path="models",
            python_model="./inference.py",
            signature=signature,
            registered_model_name=model_name,
            code_paths=[
                "./inference.py",
                "./model.py",
                "./features.py",
            ],
            artifacts={
                "model_file": f"./{model_name}.pt",
                "encoder_folder": "./encoders",
            },
        )

        # results = mlflow.evaluate(
        #     model=model_with_inference,
        #     data=val_loader,
        #     targets=val_target,
        #     metrics=["accuracy", "f1_score", "roc_auc"],
        #     model_type="pyfunc",
        # )

        # mlflow.log_metrics(results)

        client = mlflow.MlflowClient(registry_uri=registry_uri)
        client.set_registered_model_alias(
            model_name, model_alias, model_info.registered_model_version
        )
        model = mlflow.pyfunc.load_model(f"models:/{model_name}@{model_alias}")
        print(model.predict(pd.DataFrame(train_data[0:1])))


mlflow_training(
    registry_uri=registry_uri,
    epochs=5,
    categorical_cols=[
        "Marital status",
        "Application mode",
        "Application order",
        "Course",
        "Daytime/evening attendance",
        "Previous qualification",
        "Nacionality",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        "Displaced",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Scholarship holder",
    ],
)
