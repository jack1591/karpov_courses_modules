from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

from textwrap import dedent

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
    'hw_{0}_4'.format(login),
    default_args = default_args,
    start_date = datetime(2026, 1 ,1),
    schedule = timedelta(days = 1),
    catchup = False
) as dag:

    template_command = dedent(
        """
        {% for i in range(5) %}
            echo "{{ ts }}"
            echo "{{ run_id }}"
        {% endfor %}        
        """
    )

    t1 = BashOperator(
        task_id = 'template_bash_task',
        bash_command = template_command,
    )
