# The base image - Use the same version of the package `apache-airflow` that you installed via pip
FROM apache/airflow:2.7.3-python3.11
# FROM apache/airflow:latest-python3.11
# FROM apache/airflow:2.9.2-python3.11
# Why python3.11? the explanation is later below

# Set CWD inside the container
WORKDIR /project
# This will be the project folder inside the container

ENV SLUGIFY_USES_TEXT_UNIDECODE=yes


# Install requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=./requirements.txt,target=./requirements.txt \
    pip install -r ./requirements.txt

    
# RUN pip install mlflow==2.16.2
# RUN export MLFLOW_TRACKING_URI=http://localhost:8090
# RUN mlflow server -h localhost -p 8090 &

# Switch to root user
USER root

RUN curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose \
    && chmod +x /usr/local/bin/docker-compose

# Install some more CLIs
RUN apt-get update \
&& apt-get install -y --no-install-recommends vim curl git rsync unzip \
&& apt-get autoremove -y \
&& apt-get clean

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

 RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Zenml Env
RUN python3 -m venv /opt/services/zenml/venv

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=./zens.txt,target=./zens.txt \
    /opt/services/zenml/venv/bin/pip install --no-user -r ./zens.txt

# Switch to regular user airflow
USER airflow

# Run this command when we start the container
CMD ["airflow", "standalone"]