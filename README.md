# mlopls-pmldl

For "Practical Machine Learning and Deep Learning" Innopolis University course.

To run minimal deployment: FastAPI + Streamlit run:

```bash
docker-compose -f code/deployment/deployment.compose.yaml up --build
```

Generating dockerfile for model serving (check )

```bash
mlflow server -h localhost -p 8090 &
export MLFLOW_TRACKING_URI=http://localhost:8090
cd  code/deployment/models
python training.py
mlflow models generate-dockerfile --model-uri "models:/BasicModel@Champion" -d ./mlflow_api --install-mlflow
docker build -t mls mlflow_api
```

To run airflow (zenml located in DAGs):

```bash
cd services/airflow
docker-compose -f advanced_air.docker-compose.yaml up --build
```
