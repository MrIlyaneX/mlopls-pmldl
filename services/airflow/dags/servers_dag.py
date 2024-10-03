from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

from airflow.providers.docker.operators.docker import DockerOperator

# Define default arguments for the DAG
default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
}


with DAG(
    "docker_compose_deploy_up",
    default_args=default_args,
    description="Task to bring up the services using docker-compose",
    schedule_interval=None,
    catchup=False,
    tags=["tests"],
) as dag:
    docker_compose_up = BashOperator(
        task_id="docker_compose_up",
        bash_command="docker-compose -f /opt/code/deployment/deployment.compose.yaml up -d",
        dag=dag,
    )

    docker_compose_up

with DAG(
    "docker_compose_deploy_down",
    default_args=default_args,
    description="Task to bring down the services if needed",
    schedule_interval=None,
    catchup=False,
    tags=["tests"],
) as dag:
    docker_compose_down = BashOperator(
        task_id="docker_compose_down",
        bash_command="docker-compose -f /opt/code/deployment/deployment.compose.yaml down",
        dag=dag,
    )
    docker_compose_down

with DAG(
    "ls",
    default_args=default_args,
    description="Run ZenML pipeline",
    schedule_interval="*/5 * * * *",
    catchup=False,
    tags=["tests"],
) as dag:
    BashOperator(task_id="ls1", bash_command="ls /")
    BashOperator(task_id="ls2", bash_command="ls /opt")
    BashOperator(task_id="ls3", bash_command="ls /opt/services")
    BashOperator(task_id="ls4", bash_command="ls /opt/services/zenml")
    # BashOperator(task_id="ls5", bash_command="ls")
    # BashOperator(task_id="ls6", bash_command="ls")
    # BashOperator(task_id="ls7", bash_command="ls")


with DAG(
    "start_zenml_server",
    default_args=default_args,
    description="Run ZenML pipeline",
    schedule_interval="*/5 * * * *",
    catchup=False,
    tags=["tests"],
) as dag:
    BashOperator(
        task_id="zenml_01",
        bash_command="docker-compose -f /opt/services/zenml/zenml.compose.yaml up",
    )


# with DAG(
#     "zenml_pipeline_docker",
#     default_args=default_args,
#     description="Run ZenML pipeline using DockerOperator",
#     schedule_interval=None,
#     catchup=False,
#     tags=["tests"],
# ) as dag:
#     run_zenml_pipeline = DockerOperator(
#         task_id="run_zenml_pipeline",
#         image="your_zenml_image:latest",  # Custom Docker image with ZenML installed
#         command="zenml pipeline run <pipeline_name>",
#         docker_url="unix://var/run/docker.sock",
#         network_mode="bridge",
#         auto_remove=True,
#         dag=dag,
#     )

#     run_zenml_pipeline
