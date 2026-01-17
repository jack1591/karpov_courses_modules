from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# from database import SQLALCHEMY_DATABASE_URL

SQLALCHEMY_DATABASE_URL = "postgresql://postgres:password@localhost/postgres"
engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit = False, autoflush = False, bind = engine)

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "cd"}

    id = Column(Integer, primary_key = True)
    name = Column(String,name = "username")

if __name__ == "__main__":
    Base.metadata.create_all(engine)