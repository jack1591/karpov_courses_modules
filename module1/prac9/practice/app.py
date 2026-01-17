from fastapi import FastAPI, HTTPException, Depends
from datetime import date,timedelta

from psycopg2.extras import RealDictCursor
from pydantic import BaseModel
import psycopg2

class User(BaseModel):
    name: str
    surname: str
    age: int
    registration_date: date

app = FastAPI()

@app.get("/")
def say_hello(a: int, b: int):
    return a+b

@app.get("/sum_date")
def date_offset(current_date: date, offset: int):
    return current_date+timedelta(days = offset)

@app.post("/user/validate")
def validate_user(user: User):
    return f"Will add user: {user.name} {user.surname} with age {user.age}"

def get_db():
    return psycopg2.connect(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@postgres.lab.karpov.courses:6432/startml",
        cursor_factory = RealDictCursor
    )


@app.get("/user/{id}")
def get_user(id: int, db = Depends(get_db)):

    with db.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT gender, age, city
            FROM "user"
            WHERE id = {id}
            """
        )
        results = cursor.fetchone()
        if not results:
            raise HTTPException(404, "user not found")
        return results


class PostResponse(BaseModel):
    id: int
    text: str
    topic: str
    class Config:
        orm_mode = True


@app.get("/post/{id}", response_model=PostResponse)
def get_user(id: int, db = Depends(get_db)) -> PostResponse:

    with db.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT id, text, topic
            FROM "post"
            WHERE id = {id}
            """
        )
        results = cursor.fetchone()
        if not results:
            raise HTTPException(404, "user not found")
        return results

