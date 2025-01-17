from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime

Base = declarative_base()

class Competition(Base):
    __tablename__ = 'competitions'
    id = Column(Integer, primary_key=True)
    uuid = Column(String, unique=True)
    type = Column(String)
    ground_truth_html = Column(String)

    # Relationship to LeaderboardSessions
    sessions = relationship("LeaderboardSession", order_by="LeaderboardSession.id", back_populates="competition")

class LeaderboardSession(Base):
    __tablename__ = 'leaderboard_sessions'
    id = Column(Integer, primary_key=True)
    competition_uuid = Column(String, ForeignKey('competitions.uuid'))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    competition = relationship("Competition", back_populates="sessions")
    solutions = relationship("TaskSolution", back_populates="session")

class TaskSolution(Base):
    __tablename__ = 'task_solutions'
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('leaderboard_sessions.id'))
    miner_uid = Column(Integer)
    miner_hotkey = Column(String)
    validator_hotkey = Column(String)
    score = Column(Float)
    accuracy_score = Column(Float)
    quality_score = Column(Float)
    seo_score = Column(Float)
    miner_answer = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    session = relationship("LeaderboardSession", back_populates="solutions")

# Database setup
engine = create_engine('sqlite:///competition.db', echo=True)
Base.metadata.create_all(engine)

# Example function to illustrate usage
def add_competition_data():
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Create a new competition
    competition = Competition(uuid="comp-1234", type="data science")
    
    # Create a leaderboard session
    leaderboard_session = LeaderboardSession(competition_uuid=competition.uuid)
    
    # Create task solutions
    solution1 = TaskSolution(session=leaderboard_session, miner_uid=1, score=95.5)
    solution2 = TaskSolution(session=leaderboard_session, miner_uid=2, score=88.0)
    
    # Create leaderboard entry
    leaderboard_entry = LeaderboardEntry(session=leaderboard_session, average_score=91.75, max_score=95.5, min_score=88.0)
    
    # Add all to session and commit
    session.add(competition)
    session.add(leaderboard_session)
    session.add(solution1)
    session.add(solution2)
    session.add(leaderboard_entry)
    session.commit()
    session.close()

if __name__ == "__main__":
    add_competition_data()
