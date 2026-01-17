import requests
from loguru import logger

r = requests.get("http://localhost:8000/div?a=4&b=2")
logger.info(r.status_code)
logger.info(r.text)

r = requests.post(
        "http://localhost:8000/user/validate",
        json={'name': 'Artem', 'surname': 'Bezyazychnyy', 'age': 19}
    )
logger.info(f'second status code: {r.status_code}')
logger.info(f'second text: {r.text}')
