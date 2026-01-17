import psycopg2
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

class SimpleUser(BaseModel):
    name: str
    surname: str
    age: int

app = FastAPI()

@app.get("/", summary = "Just say hello")
def say_hello():
    """
    Say hello to a user
    :return:
    """
    return "hello"

@app.get("/div")
def sum_numbers(a: int, b: int):
    if b == 0:
        return "division by zero!"
    return a/b

@app.post("/user")
def print_user(name: str):
    return {"message": f"hello, {name} !"}

@app.get("/bookings/all")
def all_bookings():
    conn = psycopg2.connect(
        "postgresql://postgres:password@localhost:5432/exercises"
    )

@app.post("/user/validate")
def validate_user(user: SimpleUser):
    logger.info(user.dict())
    return 'ok'

@app.get("/error")
def show_error(a: int):
    if a==5:
        raise HTTPException(404)
    return "ok"




