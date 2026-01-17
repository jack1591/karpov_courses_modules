from datetime import datetime
from typing import Optional

from pydantic import BaseModel

class UserGet(BaseModel):
    id: int
    age: int
    city: str
    country: str
    exp_group: int
    gender: int
    os: str
    source: str

    class Config:
        orm_mode = True

class PostGet(BaseModel):
    id: int
    topic: str
    text: str
    class Config:
        orm_mode = True

class FeedGet(BaseModel):
    action: str
    user_id: int
    user: UserGet
    post_id: int
    post: PostGet
    time: datetime
    class Config:
        orm_mode = True
