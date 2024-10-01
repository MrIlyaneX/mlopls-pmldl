# mlopls-pmldl

For "Practical Machine Learning and Deep Learning" Innopolis University course.

To run airflow:

```bash
cd services/airflow

docker-compose -f advanced_air.docker-compose.yaml up --build
```

To run FastAPI + Streamlit run:

```bash
cd code/deployment

docker-compose up --build
```

Generating dockerfile for model serving

```bash
export MLFLOW_TRACKING_URI=http://localhost:8090
mlflow models generate-dockerfile --model-uri "models/basic_model@production" --env-manager virtualenv -d mlflow_api --install-mlflow
```
