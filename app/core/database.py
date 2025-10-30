from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    from sqlalchemy.orm import Session
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
