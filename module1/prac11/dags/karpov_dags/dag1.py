from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['jack.tag@mail.ru'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes = 5)
}

login = 'bezyazychnyy'

with DAG(
    'hw_{0}_1'.format(login),
    default_args = default_args,
    start_date = datetime(2026, 1, 1),
    schedule=timedelta(days=1),
    catchup = False
) as dag:
    t1 = BashOperator(
        task_id = 'pwd_task',
        bash_command = 'pwd'
    )

    def print_context(ds):
        print('Truing to print ds (logic date)')
        print(ds)

    t2 = PythonOperator(
        task_id = 'ds_print',
        python_callable = print_context
    )

    t1 >> t2