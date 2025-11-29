import os
import pickle
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.environ.get('DATABASE_URL', f"sqlite:///{os.path.join(BASE_DIR, 'data', 'embeddings.db')}")

# Create data directory
os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, index=True)
    last_name = Column(String, index=True)
    embedding = Column(LargeBinary)
    registration_date = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def save_user_embedding(first_name: str, last_name: str, embedding) -> bool:
    """Save embedding (torch tensor or numpy) as pickled bytes into DB."""
    session = SessionLocal()
    try:
        fname = first_name.lower()
        lname = last_name.lower()
        data = pickle.dumps(embedding)
        user = session.query(User).filter_by(first_name=fname, last_name=lname).first()
        if user:
            user.embedding = data
            user.registration_date = datetime.utcnow()
        else:
            user = User(first_name=fname, last_name=lname, embedding=data)
            session.add(user)
        session.commit()
        return True
    finally:
        session.close()


def load_user_embedding_db(first_name: str, last_name: str):
    session = SessionLocal()
    try:
        fname = first_name.lower()
        lname = last_name.lower()
        user = session.query(User).filter_by(first_name=fname, last_name=lname).first()
        if not user:
            return None
        return pickle.loads(user.embedding)
    finally:
        session.close()


# Initialize DB on module import
init_db()
