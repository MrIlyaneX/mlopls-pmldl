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
    "ls_directory",
    default_args=default_args,
    schedule_interval="@once",
    tags=["tests"],
) as dag:
    BashOperator(
        task_id="1",
        bash_command="ls -la /opt",
    )

    BashOperator(
        task_id="2",
        bash_command="ls -la /opt/deployment",
    )

    BashOperator(
        task_id="3",
        bash_command="ls -la /opt/deployment/api",
    )

    BashOperator(
        task_id="4",
        bash_command="ls -la /opt/deployment/app",
    )

with DAG(
    "zenml_pipeline_docker",
    default_args=default_args,
    description="Run ZenML pipeline using DockerOperator",
    schedule_interval=None,
    catchup=False,
    tags=["tests"],
) as dag:
    run_zenml_pipeline = DockerOperator(
        task_id="run_zenml_pipeline",
        image="your_zenml_image:latest",  # Custom Docker image with ZenML installed
        command="zenml pipeline run <pipeline_name>",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        auto_remove=True,
        dag=dag,
    )

    run_zenml_pipeline
