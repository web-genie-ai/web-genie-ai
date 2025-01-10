from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, UniqueConstraint, ForeignKey
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Miner(Base):
    __tablename__ = 'miners'

    id = Column(Integer, primary_key=True)
    uid = Column(Integer)
    hotkey = Column(String)
    coldkey = Column(String)
    is_registered = Column(Boolean)
    last_updated = Column(DateTime)
    __table_args__ = (
        UniqueConstraint('uid', 'hotkey', name='unique_uid_hotkey'),
    )
    

class Competition(Base):
    __tablename__ = 'competitions'

    id = Column(Integer, primary_key=True)
    uuid = Column(String)
    type = Column(String)
    ground_truth_html = Column(String)
    last_updated = Column(DateTime)


class TaskSolution(Base):
    __tablename__ = 'task_solutions'

    id = Column(Integer, primary_key=True)
    miner_uid = Column(Integer)
    miner_hotkey = Column(String)
    validator_hotkey = Column(String)
    competition_uuid = Column(String, ForeignKey('competitions.uuid'))
    score = Column(Float)
    accuracy_score = Column(Float)
    quality_score = Column(Float)
    seo_score = Column(Float)
    miner_answer = Column(String)
    last_updated = Column(DateTime)
