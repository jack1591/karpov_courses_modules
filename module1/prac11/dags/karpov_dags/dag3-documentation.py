"""
# This is documentation for dag with documentation
## Documentation - dag.doc_md = __doc__
### Title
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

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
    'hw_{0}_3'.format(login),
    default_args = default_args,
    start_date = datetime(2026, 1, 1),
    schedule=timedelta(days=1),
    catchup = False
) as dag:

    dag.doc_md = __doc__

    for i in range(10):
        bash_task = BashOperator(
            task_id = 'bash_task_{0}'.format(i),
            bash_command = f"echo {i}"
        )
        bash_task.doc_md = dedent(
            """
            # title
            ## subtitle
            **This is bash task** - *trying to finish homework*
            """
        )

    def print_task_number(task_number):
        print(f"task number is: {task_number}")

    for i in range(20):
        python_task = PythonOperator(
            task_id='python_task_{0}'.format(i),
            python_callable=print_task_number,
            op_kwargs = {'task_number': i}
        )

        python_task.doc_md = dedent(
            """
            # title
            Example of code in markdown: `task.doc_md = __doc__`
            """
        )
