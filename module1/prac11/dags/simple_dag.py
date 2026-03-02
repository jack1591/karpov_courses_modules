"""
Documentation - __doc__
"""
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.bash import BashOperator
from textwrap import dedent
from airflow.operators.python import PythonOperator

with DAG(
        'tutorial',
        default_args={
            # нужно ли ждать выполнения ДАГа для предыдущих запусков, чтобы запустить текущий
            'depends_on_past': False,
            'email': ['jack.tag@mail.ru'],
            'email_on_failure': False,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5)
        },
        description='This is tutorial DAG',
        # с какой периодичностью запускаем
        schedule=timedelta(days=1),
        # с какой даты начинаем (логическая дата)
        start_date=datetime(2026, 2, 28),
        # нужно ли запускать ДАГ для прошлого (начали с 28.02.2025 - будет 365 запусков до текущего момента)
        catchup=False,
        tags=['example']
) as dag:
    t1 = BashOperator(
        task_id='print_date',  # id задачи
        bash_command='date'  # какую команду в консоли выполнить
    )

    t2 = BashOperator(
        task_id='sleep',
        depends_on_past=False,  # переопределние параметров по умолчанию
        retries=3,  # переопределние параметров по умолчанию
        bash_command='sleep 5'
    )


    t1 >> t2