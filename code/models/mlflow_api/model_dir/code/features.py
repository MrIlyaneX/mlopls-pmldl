import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from typing import List

import joblib
import os
import json


def load_data(train_file: str, val_file: str):
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    return train_data, val_data


def fit_encoders(train_data: pd.DataFrame, target_name: str, categorial_cols: list):
    target = train_data[target_name]
    data = train_data.drop(columns=target_name)

    one_hot_encoder = OneHotEncoder(
        sparse_output=False, drop="first", handle_unknown="ignore"
    )
    scaler = StandardScaler()
    label_encoder = LabelEncoder()

    one_hot_encoder = one_hot_encoder.fit(data[categorial_cols])
    scaler = scaler.fit(data.drop(columns=categorial_cols))
    label_encoder = label_encoder.fit(target)

    return one_hot_encoder, scaler, label_encoder


def preprocess(dataset, encoder, scaler, cat_columns: List[str]):
    dataset = dataset.copy()

    data_cat = dataset[cat_columns]
    data_nc = dataset.drop(columns=cat_columns)

    data_cat_encoded = encoder.transform(data_cat)
    data_nc_scaled = scaler.transform(data_nc)

    data_cat_encoded_df = pd.DataFrame(
        data_cat_encoded, columns=encoder.get_feature_names_out(cat_columns)
    )
    data_nc_scaled_df = pd.DataFrame(data_nc_scaled, columns=data_nc.columns)

    final_data = pd.concat([data_cat_encoded_df, data_nc_scaled_df], axis=1)

    return final_data


def save_encoders(one_hot_encoder, scaler, label_encoder, cat_columns, path="encoders"):
    os.makedirs(path, exist_ok=True)

    joblib.dump(one_hot_encoder, os.path.join(path, "one_hot_encoder.pkl"))
    joblib.dump(scaler, os.path.join(path, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(path, "label_encoder.pkl"))
    with open(os.path.join(path, "cat_columns.json"), "w") as f:
        json.dump(cat_columns, f)


def load_encoders(path="encoders"):
    one_hot_encoder = joblib.load(os.path.join(path, "one_hot_encoder.pkl"))
    scaler = joblib.load(os.path.join(path, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(path, "label_encoder.pkl"))

    return one_hot_encoder, scaler, label_encoder
