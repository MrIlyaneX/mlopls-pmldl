import datetime
import logging
import os
import pickle

import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.concurrency import asynccontextmanager
from pydantic_models import StudentDataModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "best")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

model_uri = f"models/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pytorch.load_model(model_uri)

# with open("models/best.pkl", "rb") as f:
#     model = pickle.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        global model, preprocesser
        logger.info("Starting.")

        logger.info("Loading model.")

        logger.info("Model has been loaded")
    except Exception as e:
        logger.error(f"Error during starting {e}.")

    yield
    logger.info("Shutting down application.")


app = FastAPI(lifespan=lifespan)


@app.post("/main")
async def main(request: Request, data: StudentDataModel) -> None:
    logger.info(f"Received data: {data}")

    input_data = data.model_dump()

    prediction = model(torch.tensor(input_data.values, dtype=torch.float32))
    ans = preprocesser.reverse_label_transform(prediction.tolist())

    print(type(ans))
    return {"prediction": ans.tolist()}
