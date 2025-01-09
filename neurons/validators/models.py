from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Miner(Base):
    __tablename__ = 'miners'
    
    id = Column(Integer, primary_key=True)
    uid = Column(Integer)
    reward = Column(Float)
    competition_type = Column(String)
    timestamp = Column(DateTime)
