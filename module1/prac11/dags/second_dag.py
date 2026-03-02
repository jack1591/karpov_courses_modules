"""
Documentation - __doc__
"""
from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.bash import BashOperator
from textwrap import dedent

with DAG(
    'my_second_dag',
    default_args = {
        # нужно ли ждать выполнения ДАГа для предыдущих запусков, чтобы запустить текущий
        'depends_on_past': False,
        'email': ['jack.tag@mail.ru'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes = 5)
    },
    description = 'This is tutorial DAG',
    # с какой периодичностью запускаем
    schedule = timedelta(days = 1),
    # с какой даты начинаем (логическая дата)
    start_date = datetime(2026, 2, 28),
    # нужно ли запускать ДАГ для прошлого (начали с 28.02.2025 - будет 365 запусков до текущего момента)
    catchup = False,
    tags = ['example']
) as dag:
    t1 = BashOperator(
        task_id = 'print_date', #id задачи
        bash_command = 'date' #какую команду в консоли выполнить
    )

    t1.doc_md = dedent(
        """
        ### Task Documentation  
        This is documentation for single task
        """
    )

    t2 = BashOperator(
        task_id = 'sleep',
        depends_on_past = False, # переопределние параметров по умолчанию
        retries = 3, # переопределние параметров по умолчанию
        bash_command = 'date'
    )

    dag.doc_md = __doc__

    # Jinja шаблон
    # ds - дата, для которой выполняем задачу
    # macros.ds_add - добавляем к текущей дате какое то значение (7 дней)
    templated_command = dedent(
        """
        {% for i in range(5) %}
            echo "{{ ds }}"
            echo "{{ macros.ds_add(ds, 7)}}"
        {% endfor %}
        """
    )

    t3 = BashOperator(
        task_id = 'templated', #id задачи
        depends_on_past = False,
        bash_command = templated_command #какую команду в консоли выполнить
    )

    from airflow.operators.python import PythonOperator

    def print_context(**kwargs):
        # ds - логическая дата DAG в виде строки
        # **kwargs - словарь аргументов, которые мы передаем в функцию
        print(kwargs)

        # выводим значение параметра ds - он является ключом в словаре передаваемых аргументов
        print(kwargs['ds'])
        return 'Test python operator'


    run_this = PythonOperator(
        task_id='test_python_operator',
        python_callable=print_context
    )

    # в Python-оператор можно передавать аргументы функции при вызове
    def my_sleeping_function(random_base):
        import time
        time.sleep(random_base)

    t1 >> [t2, t3] >> run_this

    for i in range(5):
        my_task = PythonOperator(
            task_id='op_kwargs_task_' + str(i),
            python_callable=my_sleeping_function,
            op_kwargs={'random_base': float(i) / 2}
        )

        run_this >> my_task

    # Пример передачи сскрипта в BashOperator
    new_task = BashOperator(
        task_id='task_with_sh',
        bash_command='/opt/airflow/scripts/my_script.sh ',  # Пробел после названия !
        dag=dag,  # задача принадлежит нашему графу
        #env={'DATA_INTERVAL_START': '{{ ds }}'}  # переменные окружения
    )
