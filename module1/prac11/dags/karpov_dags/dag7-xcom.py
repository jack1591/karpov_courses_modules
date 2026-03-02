from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

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
    'hw_{0}_7'.format(login),
    default_args = default_args,
    start_date = datetime(2026, 1, 1),
    schedule = timedelta(days = 1),
    catchup = False
) as dag:

    def push_data_xcom(ti):
        ti.xcom_push(
            key = 'sample_xcom_key',
            value = 'xcom test'
        )

    def get_data_xcom(ti):
        data = ti.xcom_pull(
            key = 'sample_xcom_key',
            task_ids = 'push_task'
        )
        print(data)

    push_task = PythonOperator(
        task_id = 'push_task',
        python_callable = push_data_xcom
    )

    pull_task = PythonOperator(
        task_id = 'pull_task',
        python_callable = get_data_xcom
    )

    push_task >> pull_task
