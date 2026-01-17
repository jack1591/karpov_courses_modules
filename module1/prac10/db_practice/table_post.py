from database import Base, SessionLocal
from sqlalchemy import Column, Integer, String

class Post(Base):
    __tablename__ = "post"
    id = Column(Integer, primary_key = True)
    topic = Column(String)
    text = Column(String)

if __name__ == "__main__":
    session = SessionLocal()
    answer = []
    for elem in session.query(Post).filter(Post.topic == "business").order_by(Post.id.desc()).limit(10).all():
        answer.append(elem.id)
    print(answer)
