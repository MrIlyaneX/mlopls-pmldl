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
    "docker_compose_up_example",
    default_args=default_args,
    description="An example DAG to run docker-compose up for api and app services",
    schedule_interval=None,
    catchup=False,
    tags=["tests"],
) as dag:
    # Task to bring up the services using docker-compose
    docker_compose_up = BashOperator(
        task_id="docker_compose_up",
        bash_command="docker-compose -f /opt/deployment/compose.yaml up -d",
        dag=dag,
    )

    # Task to bring down the services if needed
    docker_compose_down = BashOperator(
        task_id="docker_compose_down",
        bash_command="docker-compose -f /opt/deployment/compose.yaml down",
        dag=dag,
    )

    # Define task dependencies (optional)
    docker_compose_up >> docker_compose_down

with DAG(
    dag_id="docker_build_example",
    default_args=default_args,
    schedule_interval=None,
    tags=["tests"],
) as dag:
    # Build Docker image inside the DAG
    build_image = BashOperator(
        task_id="build_api_docker_image",
        bash_command="docker build -t deployment_sreamlit_air:latest /Users/ilia/learning_ds/F24/mlopls-pmldl/code/deployment/app",
        dag=dag,
    )

    build_image

with DAG(
    dag_id="docker_run_example",
    default_args=default_args,
    schedule_interval=None,
    tags=["tests"],
) as dag:
    run_docker_image = DockerOperator(
        task_id="run_api_docker_image",
        container_name="deployment-sreamlit",
        image="deployment_sreamlit_air:latest",  # Use the image name from above
        # command="python /app/main.py",  # Command to run in the container
        # command="uvicorn app:app --host 0.0.0.0 --port 8000",
        command="streamlit run app.py",
        # docker_url="unix://var/run/docker.sock",  # To communicate with Docker daemon
        network_mode="bridge",
        environment={"FASTAPI_URL": "http://fastapi:8000/main"},
        auto_remove="success",
        mount_tmp_dir=False,
        # volumes=[
        #     "/Users/ilia/learning_ds/F24/mlopls-pmldl/code/deployment/app:/app/"
        # ],
    )

    run_docker_image
