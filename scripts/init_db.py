from app.models.base import Base
from app.models import models
from app.core.database import engine

print("Creating database tables")
Base.metadata.create_all(bind=engine)
print("Tables created successfully")
