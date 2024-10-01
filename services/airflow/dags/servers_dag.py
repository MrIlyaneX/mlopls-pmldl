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
        bash_command="docker-compose -f /opt/code/deployment/compose.yaml up -d",
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
        bash_command="docker-compose -f /opt/code/deployment/compose.yaml down",
        dag=dag,
    )
    docker_compose_down
