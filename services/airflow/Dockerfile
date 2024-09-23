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

# Switch to root user
USER root

RUN curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose \
    && chmod +x /usr/local/bin/docker-compose

# Install some more CLIs
RUN apt-get update \
&& apt-get install -y --no-install-recommends vim curl git rsync unzip \
&& apt-get autoremove -y \
&& apt-get clean

# EXPOSE 8080

# Switch to regular user airflow
USER airflow

# Run this command when we start the container
CMD ["airflow", "standalone"]