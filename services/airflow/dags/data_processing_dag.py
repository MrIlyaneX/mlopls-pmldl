# from pendulum import datetime
# import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from airflow.utils.dates import days_ago

from airflow.providers.docker.operators.docker import DockerOperator

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
}


with DAG(
    "run_local_zenml",
    default_args=default_args,
    description="Run ZenML pipeline",
    schedule_interval="*/5 * * * *",
    catchup=False,
    tags=["tests"],
) as dag:
    run_pipe = BashOperator(
        task_id="run_pipeline",
        bash_command="""
            if ! ls /opt/data | grep -q processed; then \
                mkdir /opt/data/processed; \
            fi && \
            source /opt/services/zenml/venv/bin/activate && \
            cd /opt/services/zenml && \
            zenml init && \
            zenml down && \
            zenml up && \
            if ! zenml artifact-store list | grep -q artifacts; then \
                zenml artifact-store register artifacts --flavor=local; \
            fi && \
            if ! zenml stack list | grep -q dev_stack; then \
                zenml stack register dev_stack -o default -a artifacts; \
            fi && \
            zenml stack set dev_stack && \
            python /opt/services/zenml/pipelines/training_pipeline.py -training_pipeline
        """,
    )

    run_pipe
