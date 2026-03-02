from datetime import datetime, timedelta
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

def get_variable():
    from airflow.models import Variable
    is_prod = Variable.get('is_prod')
    return is_prod

def get_connection():
    from airflow.hooks.base import BaseHook

    connection = BaseHook.get_connection('postgres_main')
    conn_password = connection.password
    conn_login = connection.login
    return conn_password, conn_login

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['jack.tag@mail.ru'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes = 5)
}

with DAG(
    'testing_variables_dag',
    default_args = default_args,
    start_date = datetime(2026, 1, 1),
    schedule = timedelta(days = 1),
    catchup = False
) as dag:
    t1 = PythonOperator(
        task_id = 'get_variable',
        python_callable = get_variable
    )

    t2 = PythonOperator(
        task_id = 'get_connection',
        python_callable = get_connection
    )

    t1 >> t2