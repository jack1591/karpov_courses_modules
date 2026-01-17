from typing import List

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import desc, func
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import SessionLocal
from schema import UserGet, PostGet, FeedGet

from table_user import User
from table_post import Post
from table_feed import Feed

import os
import pickle
from catboost import CatBoostClassifier

import pandas as pd
from sqlalchemy import create_engine


app = FastAPI()

def get_db():
    with SessionLocal() as db:
        return db

@app.get("/users/{limit}", response_model=List[UserGet])
def first_users(limit: int = 10, db: Session = Depends(get_db)) -> List[UserGet]:
    result = (db.query(User).limit(limit).all())
    if not result:
        raise HTTPException(404)
    return result

@app.get("/user/{id}", response_model=UserGet)
def get_user(id: int, db: Session = Depends(get_db))->UserGet:
    result = db.query(User).filter(User.id==id).first()
    if not result:
        raise HTTPException(404, "user not found!")
    return result

@app.get("/post/{id}", response_model = PostGet)
def get_post(id: int, db: Session = Depends(get_db)) -> PostGet:
    result = db.query(Post).filter(Post.id==id).first()
    if not result:
        raise HTTPException(404, "post not found!")
    return result

@app.get("/user/{id}/feed", response_model = List[FeedGet])
def get_user_feed(id: int, limit: int = 10, db: Session = Depends(get_db)) -> List[FeedGet]:
    result = (db.query(Feed)
              .filter(Feed.user_id == id)
              .order_by(desc(Feed.time))
              .limit(limit)
              .all())
    if not result:
        raise HTTPException(200, [])
    return result

@app.get("/post/{id}/feed", response_model = List[FeedGet])
def get_user_feed(id: int, limit: int = 10, db: Session = Depends(get_db)) -> List[FeedGet]:
    result = (db.query(Feed)
              .filter(Feed.post_id == id)
              .order_by(desc(Feed.time))
              .limit(limit)
              .all())
    if not result:
        raise HTTPException(200, [])
    return result

@app.get("/post/recommendations/", response_model = List[PostGet])
def top_posts(id: int, limit: int = 10, db: Session = Depends(get_db)):
    result = (db.query(Post)
              .select_from(Feed)
              .filter(Feed.action == 'like').join(Post, Post.id == Feed.post_id )
              .group_by(Post.id)
              .order_by(desc(func.count(Post.id)))
              .limit(limit)
              .all())

    if not result:
        raise HTTPException(404)
    return result


@app.get("/")
def read_root():
    return {"Hello": "World!"}
