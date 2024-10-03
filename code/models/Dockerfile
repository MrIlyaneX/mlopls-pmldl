# Build an image that can serve mlflow models.
FROM python:3.11.9-slim

RUN apt-get -y update && apt-get install -y --no-install-recommends nginx
RUN apt-get update && apt-get install -y git && apt-get clean

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# RUN python3 -m venv /opt/code/models/venv

# RUN --mount=type=cache,target=/root/.cache/pip \
#     --mount=type=bind,source=./requirements.txt,target=./requirements.txt \
#     /opt/code/models/venv/bin/pip install --no-user -r ./requirements.txt

# ENV PATH="/opt/code/models/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=./requirements.txt,target=./requirements.txt \
    pip install --no-user -r ./requirements.txt

WORKDIR /opt/code/models

# Install MLflow
RUN pip install mlflow==2.16.2

# Copy model to image and install dependencies
COPY . /opt/code/models

ENV MLFLOW_DISABLE_ENV_CREATION=True
ENV ENABLE_MLSERVER=False
ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent"

RUN chmod +x /opt/code/models/init.sh

EXPOSE 8090
CMD ["bash", "/opt/code/models/init.sh"]

# mlflow models generate-dockerfile --model-uri "models:/BasicModel@Champion" -d code/models/mlflow_api --install-mlflow

# docker build -t mlflow_service mlflow_api
# docker run --rm -p 5152:8010 mlflow_service

# mlflow server -h localhost -p 8090

# RUN cd /opt/code/models
# RUN  mlflow server -h localhost -p 8090 &

# RUN python training.py

# RUN mlflow models generate-dockerfile --model-uri "models:/BasicModel@Champion" -d opt/mlflow_api --install-mlflow
# RUN docker build -t mlflow_service mlflow_api