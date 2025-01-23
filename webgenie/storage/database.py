from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import DeclarativeBase
# Create the database engine
engine = create_engine('sqlite:///webgenie-validator.db', echo=False)

# Create the session maker
Session = sessionmaker(engine)

# Create the base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass

