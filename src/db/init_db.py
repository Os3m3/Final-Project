from .base import Base, engine
from . import models  # noqa: F401  (import so metadata sees models)

if __name__ == "__main__":
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)
    print("Done. Check your database for 'users' and 'attendance' tables.")