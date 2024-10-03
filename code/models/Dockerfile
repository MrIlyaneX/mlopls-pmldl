# Build an image that can serve mlflow models.
FROM python:3.11.9-slim

RUN apt-get -y update && apt-get install -y --no-install-recommends nginx

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=./requirements.txt,target=./requirements.txt \
    uv pip install -r ./requirements.txt
