import json
from datetime import datetime, timedelta
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
import random
url = 'https://covidtracking.com/api/v1/states'
state = 'wa'

# сохраним данные
def get_testing_increase(state, ti):
    #ti - task instance

    #res = requests.get(url + '{0}/current.json'.format(state))
    #testing_increase = json.loads(res.text)['totalTestResultsIncrease']

    testing_increase = random.randint(1000, 10000)
    ti.xcom_push(
        key = 'testing_increase',
        value = testing_increase
    )

# получим данные
def analyze_testing_increases(state, ti):
    testing_inreases = ti.xcom_pull(
        key = 'testing_increase',
        task_ids = 'get_testing_increase_data_{0}'.format(state)
    )
    print('Testing increases for {0}:'.format(state), testing_inreases)

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
    'xcom_dag',
    start_date = datetime(2021, 1, 1),
    schedule = timedelta(minutes = 30),
    default_args = default_args,
    catchup = False
) as dag:
    opr_get_data = PythonOperator(
        task_id = 'get_testing_increase_data_{0}'.format(state),
        python_callable = get_testing_increase,
        op_kwargs = {'state': state}
    )

    opr_analyze_data = PythonOperator(
        task_id='analyze_data',
        python_callable=analyze_testing_increases,
        op_kwargs={'state': state}
    )

    opr_get_data >> opr_analyze_data

