from sqlalchemy import Column, Integer, String, func, desc

from database import Base, SessionLocal


class User(Base):
    __tablename__='user'
    id = Column(Integer, primary_key = True)
    age = Column(Integer)
    city = Column(String)
    country = Column(String)
    exp_group = Column(Integer)
    gender = Column(Integer)
    os = Column(String)
    source = Column(String)

if __name__ == "__main__":
    session = SessionLocal()
    query =  (session.query(User.country, User.os, func.count(User.id).label("count"))
     .filter(User.exp_group==3)
     .group_by(User.country, User.os)
     .order_by(desc("count"))
     .having(func.count(User.id)>100)
     .all())
    answer = []
    for country,os,count in query:
        answer.append((country,os,count))

    print(answer)