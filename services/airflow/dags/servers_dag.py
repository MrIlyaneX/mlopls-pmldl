from time import sleep
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
}


with DAG(
    "run_local_zenml",
    default_args=default_args,
    description="Run ZenML pipeline",
    schedule_interval=None,
    catchup=False,
    tags=["tests"],
) as run_local_zenml:
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


with DAG(
    "docker_compose_deploy_up",
    default_args=default_args,
    description="Task to bring up the services using docker-compose",
    schedule_interval=None,
    catchup=False,
    tags=["tests"],
) as docker_compose_deploy_up:
    docker_compose_up = BashOperator(
        task_id="docker_compose_up",
        bash_command="docker-compose -f /opt/code/deployment/deployment.compose.yaml up -d",
        dag=docker_compose_deploy_up,
    )

    docker_compose_up

with DAG(
    "docker_compose_deploy_down",
    default_args=default_args,
    description="Task to bring down the services if needed",
    schedule_interval=None,
    catchup=False,
    tags=["tests"],
) as docker_compose_deploy_down:
    docker_compose_down = BashOperator(
        task_id="docker_compose_down",
        bash_command="docker-compose -f /opt/code/deployment/deployment.compose.yaml down",
        dag=docker_compose_deploy_up,
    )
    docker_compose_down


with DAG(
    "run_training",
    default_args=default_args,
    description="Run training with MLflow and build model container for deployment",
    schedule_interval=None,
    catchup=False,
    tags=["tests"],
) as run_training:
    train = BashOperator(
        task_id="run_training",
        bash_command="""
            bash /opt/code/models/init.sh && \
            cd /opt/code/models && \
            docker build -t mls mlflow_api
            """,
        dag=run_training,
    )
    train


with DAG(
    "master_dag",
    default_args=default_args,
    description="Master DAG",
    schedule_interval="*/5 * * * *",
    catchup=False,
    tags=["tests"],
) as master_dag:
    trigger_run_local_zenml = TriggerDagRunOperator(
        task_id="trigger_run_local_zenml",
        trigger_dag_id="run_local_zenml",
    )

    trigger_docker_compose_deploy_up = TriggerDagRunOperator(
        task_id="trigger_docker_compose_deploy_up",
        trigger_dag_id="run_training",
    )

    trigger_docker_compose_deploy_down = TriggerDagRunOperator(
        task_id="trigger_docker_compose_deploy_down",
        trigger_dag_id="docker_compose_deploy_up",
    )

    trigger_run_training = TriggerDagRunOperator(
        task_id="trigger_run_training",
        trigger_dag_id="docker_compose_deploy_down",
    )

    wait_timer_task = PythonOperator(
        task_id="waiting",
        python_callable=lambda: sleep(300),
        dag=master_dag,
    )

    (
        trigger_run_local_zenml
        >> trigger_run_training
        >> trigger_docker_compose_deploy_up
        >> wait_timer_task
        >> trigger_docker_compose_deploy_down
    )
