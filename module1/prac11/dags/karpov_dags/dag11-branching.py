from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator

from airflow.operators.empty import EmptyOperator

from datetime import datetime, timedelta
from airflow.models import Variable

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

with (DAG(
    'hw_{0}_11'.format(login),
    default_args = default_args,
    start_date = datetime(2026, 1, 1),
    schedule = timedelta(days = 1),
    catchup = False
) as dag):
    start_task = EmptyOperator(
        task_id = 'before_branching'
    )

    def choose_branch():
        is_startml = Variable.get('is_startml')
        if (is_startml == 'True'):
            return 'startml_desc'
        return 'not_startml_desc'

    def startml_desc():
        print('StartML is a starter course for ambitious people')

    def not_startml_desc():
        print('Not a startML course, sorry')

    branching_task = BranchPythonOperator(
        task_id = 'determine_course',
        python_callable = choose_branch
    )

    startml_desc_task = PythonOperator(
        task_id = 'startml_desc',
        python_callable = startml_desc
    )

    not_startml_desc_task = PythonOperator(
        task_id = 'not_startml_desc',
        python_callable = not_startml_desc
    )

    finish_task = EmptyOperator(
        task_id='after_branching',
        trigger_rule='none_failed_min_one_success'
    )

    start_task >> branching_task
    branching_task >> [startml_desc_task, not_startml_desc_task]
    startml_desc_task >> finish_task
    not_startml_desc_task >> finish_task
