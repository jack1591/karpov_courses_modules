from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

def func():
    postgres = PostgresHook(postgres_conn_id="startml_feed")
    with postgres.get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
            SELECT user_id, count(action) AS count
            FROM feed_action
            WHERE action='like'
            GROUP BY user_id
            ORDER BY count desc  
            LIMIT 1                                   
            """)
            result = cursor.fetchone()
            return result

login = 'bezyazychnyy'

default_args = {
            'depends_on_past': False,
            'email': ['airflow@example.com'],
            'email_on_failure': False,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(days=1),
        }

with DAG(
        'hw_{0}_9'.format(login),
        default_args = default_args,
        start_date = datetime(2026, 1, 1),
        schedule = timedelta(days=1),
        catchup = True,
) as dag:
    get_user_task = PythonOperator(
        task_id='get_user_task',
        python_callable=func,
    )