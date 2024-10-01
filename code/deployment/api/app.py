import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.concurrency import asynccontextmanager
from pydantic_models import StudentDataModel

import requests


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "basic_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow:8090")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        global model, preprocesser
        logger.info("Starting.")
        logger.info(f"Using MLflow model: {MODEL_NAME}@{MODEL_ALIAS}")
    except Exception as e:
        logger.error(f"Error during starting {e}.")

    yield
    logger.info("Shutting down application.")


app = FastAPI(lifespan=lifespan)


@app.post("/main")
async def main(request: Request, data: StudentDataModel) -> None:
    logger.info(f"\n\nReceived data: {data}\n\n")

    # input_data = data.model_dump()

    input_payload = {
        "inputs": [
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.14744850993156433,
                -0.3712688982486725,
                -0.6210671663284302,
                -0.0816706046462059,
                -0.16070155799388885,
                0.06489204615354538,
                0.7546852231025696,
                0.6776396632194519,
                0.5706744194030762,
                -0.14188984036445618,
                -0.1467650979757309,
                0.04092109203338623,
                0.21853511035442352,
                0.3580314815044403,
                0.6804752349853516,
                -0.1351272612810135,
                0.8968483805656433,
                -1.092515230178833,
                0.3868410587310791,
            ]
        ]
    }

    prediction = None

    try:
        response = requests.post(
            f"{MLFLOW_URL}/invocations",
            json=input_payload,
        )

        if response.status_code == 200:
            prediction = response.json()
            logger.info("Prediction:", prediction)
        else:
            logger.warning(f"Error: Received status code {response.status_code}")
            logger.warning("Response content:", response.text)

    except Exception as e:
        logger.error(f"An error occurred while making the request: {e}")

    return {"prediction": prediction}
