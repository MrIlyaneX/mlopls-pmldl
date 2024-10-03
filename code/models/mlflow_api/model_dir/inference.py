from typing import List

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import torch

from mlflow.pyfunc import PythonModel
from mlflow.models import set_model
import torch
from torch import nn

from model import BasicNet

import json
import os
import joblib


class ModelInferenceInterface(PythonModel):
    def __init__(
        self,
        model: nn.Module | None = None,
        cat_columns: List[str] | None = None,
    ) -> None:
        super().__init__()
        self.one_hot_encoder = None
        self.scaler = None
        self.label_encoder = None
        self.cat_columns = cat_columns
        self.model = model if model else BasicNet(271)

    def __process(self, data) -> pd.DataFrame:
        data_cat = data[self.cat_columns]
        data_nc = data.drop(columns=self.cat_columns)

        data_cat_encoded = self.one_hot_encoder.transform(data_cat)
        data_nc_scaled = self.scaler.transform(data_nc)

        data_cat_encoded_df = pd.DataFrame(
            data_cat_encoded,
            columns=self.one_hot_encoder.get_feature_names_out(self.cat_columns),
        )
        data_nc_scaled_df = pd.DataFrame(data_nc_scaled, columns=data_nc.columns)

        final_data = pd.concat([data_cat_encoded_df, data_nc_scaled_df], axis=1)

        return final_data

    def predict(self, context, model_input: pd.DataFrame, params=None):
        data = self.__process(model_input)
        with torch.no_grad():
            outputs = self.model(torch.tensor(data.values, dtype=torch.float32))
            predictions = outputs.argmax(dim=1).numpy()

        return self.label_encoder.inverse_transform(predictions)

    def load_context(self, context):
        encoders = context.artifacts["encoder_folder"]
        model_ = context.artifacts["model_file"]

        self.one_hot_encoder = joblib.load(
            os.path.join(encoders, "one_hot_encoder.pkl")
        )
        self.scaler = joblib.load(os.path.join(encoders, "scaler.pkl"))
        self.label_encoder = joblib.load(os.path.join(encoders, "label_encoder.pkl"))

        with open(os.path.join(encoders, "cat_columns.json"), "r") as f:
            self.cat_columns = json.load(f)

        self.model.load_state_dict(torch.load(model_))
        self.model.eval()

        # print("Woops2")
        # print(self.one_hot_encoder, self.scaler, self.label_encoder)
        # print(self.model)


set_model(ModelInferenceInterface())
