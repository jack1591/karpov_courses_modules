from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

login = 'bezyazychnyy'

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
    'hw_{0}_10'.format(login),
    default_args = default_args,
    start_date = datetime(2026, 1, 1),
    schedule = timedelta(days = 1),
    catchup = False
) as dag:

    def get_variable():
        from airflow.models import Variable
        is_startml = Variable.get('is_startml')
        print(is_startml)
        return is_startml

    varialble_task = PythonOperator(
        task_id = 'get_variable_task',
        python_callable = get_variable
    )