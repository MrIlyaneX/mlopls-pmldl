import logging
import os

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.concurrency import asynccontextmanager
from sqlalchemy import Null
from pydantic_models import StudentDataModel

from typing import Dict

import requests


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "BasicModel")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "Champion")
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
async def main(request: Request, data: StudentDataModel):
    logger.info(f"\n\nReceived data: {data}\n\n")

    input_data = data.model_dump()

    df = pd.DataFrame([input_data])

    original_columns = [
        "Marital status",
        "Application mode",
        "Application order",
        "Course",
        "Daytime/evening attendance",
        "Previous qualification",
        "Previous qualification (grade)",
        "Nacionality",
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
        "Admission grade",
        "Displaced",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Scholarship holder",
        "Age at enrollment",
        "International",
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ]

    df.columns = original_columns

    input_payload = {
        "dataframe_split": {
            "columns": df.columns.tolist(),
            "data": df[0:1].values.tolist(),
        }
    }

    logger.info(input_payload)

    try:
        response = requests.post(
            f"{MLFLOW_URL}/invocations",
            json=input_payload,
        )

        if response.status_code == 200:
            prediction = response.json()
            logger.info("Prediction:", prediction)
            return {"prediction": prediction}
        else:
            logger.warning(f"Error: Received status code {response.status_code}")
            logger.warning("Response content:", response.text)

    except Exception as e:
        logger.error(f"An error occurred while making the request: {e}")

    return {"prediction": None}
