# -----------------------------------------------
#
# All-in-one preprocessing, currently unused due to deployment complications and time constrains
#
# -----------------------------------------------


# from typing import Annotated, Any, Tuple
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
# from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
# from zenml import ArtifactConfig, save_artifact, step
# from zenml.integrations import SklearnIntegration, PandasIntegration


# @step(enable_cache=False)
# def training_data_loader(train_data_path: str, target_name: str) -> Tuple[
#     Annotated[pd.DataFrame, ArtifactConfig(name="features", tags=["data_preparation"])],
#     Annotated[pd.DataFrame, ArtifactConfig(name="target", tags=["data_preparation"])],
# ]:
#     df_train = pd.read_csv(train_data_path, index_col=0)
#     target = df_train[target_name].to_frame()
#     data = df_train.drop(target_name, axis=1)
#     return data, target


# @step(enable_cache=False)
# def test_data_loader(
#     test_data_path: str,
# ) -> Annotated[
#     pd.DataFrame, ArtifactConfig(name="features", tags=["data_preparation"])
# ]:
#     df_test = pd.read_csv(test_data_path, index_col=0)
#     return df_test


# @step(enable_cache=False)
# def fit_encoders_and_scalers(
#     data: pd.DataFrame, target: pd.Series, categorial_cols: list
# ) -> Tuple[
#     Annotated[
#         OneHotEncoder, ArtifactConfig(name="onehot_encoder", tags=["preprocessers"])
#     ],
#     Annotated[
#         StandardScaler, ArtifactConfig(name="std_scaler", tags=["preprocessers"])
#     ],
#     Annotated[
#         LabelEncoder,
#         ArtifactConfig(name="label_encoder", tags=["preprocessers"]),
#     ],
# ]:
#     SklearnIntegration.activate()
#     PandasIntegration.activate()

#     one_hot_encoder = OneHotEncoder(
#         sparse_output=False, drop="first", handle_unknown="ignore"
#     )
#     one_hot_encoder.fit(data[categorial_cols])

#     scaler = StandardScaler()
#     scaler.fit(data.drop(columns=categorial_cols))

#     label_encoder = LabelEncoder()
#     label_encoder.fit(target)

#     save_artifact(categorial_cols, name="categorial_cols")

#     return one_hot_encoder, scaler, label_encoder


# @step(enable_cache=False)
# def preprocess_data(
#     data: pd.DataFrame,
#     categorial_cols: list,
#     one_hot_encoder: OneHotEncoder,
#     scaler: StandardScaler,
# ) -> Annotated[
#     pd.DataFrame,
#     ArtifactConfig(name="input_features", tags=["data_preparation", "inference"]),
# ]:
#     data_cat = data[categorial_cols]
#     data_cat_encoded = one_hot_encoder.transform(data_cat)

#     data_num = data.drop(columns=categorial_cols)
#     data_num_scaled = scaler.transform(data_num)

#     return pd.DataFrame(np.concatenate([data_cat_encoded, data_num_scaled], axis=1))


# @step(enable_cache=False)
# def preprocess_target(
#     target: pd.Series,
#     label_encoder: LabelEncoder,
# ) -> Annotated[
#     np.ndarray, ArtifactConfig(name="input_target", tags=["data_preparation"])
# ]:
#     return label_encoder.transform(target)


# @step(enable_cache=False)
# def split_data(
#     processed_train_data: pd.DataFrame, test_size: float = 0.2, seed: int = 42
# ) -> Tuple[
#     Annotated[
#         np.ndarray, ArtifactConfig(name="train_indeces", tags=["data_preparation"])
#     ],
#     Annotated[
#         np.ndarray, ArtifactConfig(name="validation_indeces", tags=["data_preparation"])
#     ],
# ]:
#     np.random.seed(seed)
#     indices = np.random.permutation(len(processed_train_data))
#     test_split = int(len(processed_train_data) * test_size)
#     train_indices, test_indices = indices[test_split:], indices[:test_split]
#     return train_indices, test_indices


# @step(enable_cache=False)
# def create_train_torch_dataloader(
#     processed_train_data: pd.DataFrame,
#     target_encoded: np.ndarray,
#     train_idx: np.ndarray,
#     val_idx: np.ndarray,
#     batch_size: int = 128,
#     save_torch_loaders: bool = True,
#     data_path: str = "data/processed",
# ) -> Tuple[
#     Annotated[
#         DataLoader, ArtifactConfig(name="train_dataloader", tags=["data_preparation"])
#     ],
#     Annotated[
#         DataLoader,
#         ArtifactConfig(name="validation_dataloader", tags=["data_preparation"]),
#     ],
# ]:
#     X = torch.tensor(processed_train_data.values, dtype=torch.float32)
#     y = torch.tensor(target_encoded, dtype=torch.int64)

#     processed_dataset = TensorDataset(X, y)
#     train_sampler = SubsetRandomSampler(train_idx)
#     val_sampler = SubsetRandomSampler(val_idx)

#     train_loader = DataLoader(
#         processed_dataset, batch_size=batch_size, sampler=train_sampler
#     )
#     val_loader = DataLoader(
#         processed_dataset, batch_size=batch_size, sampler=val_sampler
#     )

#     if save_torch_loaders:
#         torch.save(train_loader, f"{data_path}/train_loader.pt")
#         torch.save(val_loader, f"{data_path}/val_loader.pt")

#     return train_loader, val_loader


# @step(enable_cache=False)
# def create_test_torch_dataloader(
#     test_data: pd.DataFrame,
#     batch_size: int = 128,
#     save_torch_loaders: bool = True,
#     data_path: str = "data/processed",
# ) -> Annotated[
#     DataLoader,
#     ArtifactConfig(name="test_dataloader", tags=["data_preparation"]),
# ]:
#     X_test = torch.tensor(test_data.values, dtype=torch.float32)
#     test_dataset = TensorDataset(X_test)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     if save_torch_loaders:
#         torch.save(test_loader, f"{data_path}/test_loader.pt")

#     return test_loader


# @step(enable_cache=False)
# def tensor_converter(
#     data: pd.DataFrame,
# ) -> Annotated[
#     torch.Tensor, ArtifactConfig(name="tensor_converter", tags=["data_preparation"])
# ]:
#     return torch.tensor(data.values, dtype=torch.float32)

# @pipeline(enable_cache=False)
# def training_pipeline(
#     train_data_path: str = "opt/data/raw/train.csv",
# ) -> Tuple[
#     Annotated[
#         DataLoader, ArtifactConfig(name="train_dataloader", tags=["data_preparation"])
#     ],
#     Annotated[
#         DataLoader,
#         ArtifactConfig(name="validation_dataloader", tags=["data_preparation"]),
#     ],
# ]:
#     data, target = training_data_loader(train_data_path, target_name="Target")

#     categorial_cols = [
#         "Marital status",
#         "Application mode",
#         "Application order",
#         "Course",
#         "Daytime/evening attendance",
#         "Previous qualification",
#         "Nacionality",
#         "Mother's qualification",
#         "Father's qualification",
#         "Mother's occupation",
#         "Father's occupation",
#         "Displaced",
#         "Educational special needs",
#         "Debtor",
#         "Tuition fees up to date",
#         "Gender",
#         "Scholarship holder",
#     ]

#     one_hot_encoder, scaler, label_encoder = fit_encoders_and_scalers(
#         data, target, categorial_cols
#     )
#     processed_data = preprocess_data(data, categorial_cols, one_hot_encoder, scaler)

#     target_encoded = preprocess_target(target, label_encoder)

#     train_idx, val_idx = split_data(processed_data)

#     train_loader, val_loader = create_train_torch_dataloader(
#         processed_data, target_encoded, train_idx, val_idx, data_path="/opt/data"
#     )

#     return train_loader, val_loader
