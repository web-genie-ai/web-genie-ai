from sqlalchemy import Column, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column
from datetime import datetime
from database import Base, engine

class Neuron(Base):
    __tablename__ = "neurons"
    id: Mapped[int] = mapped_column(primary_key=True)
    coldkey: Mapped[str]
    hotkey: Mapped[str] = mapped_column(index=True)

class Competition(Base):
    __tablename__ = "competitions"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(index=True)

    # Relationships
    sessions: Mapped[list["LeaderboardSession"]] = relationship(back_populates="competition")

class LeaderboardSession(Base):
    __tablename__ = "leaderboard_sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"), index=True)

    # Relationships
    competition: Mapped["Competition"] = relationship(back_populates="sessions")
    challenges: Mapped[list["Challenge"]] = relationship(back_populates="session")

class Challenge(Base):
    __tablename__ = "challenges"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("leaderboard_sessions.id"), index=True)
    ground_truth_html: Mapped[str]

    # Relationships
    session: Mapped["LeaderboardSession"] = relationship(back_populates="challenges")
    solutions: Mapped[list["TaskSolution"]] = relationship(back_populates="challenge")


class Judgement(Base):
    __tablename__ = "judgements"
    id: Mapped[int] = mapped_column(primary_key=True)
    validator_id: Mapped[int] = mapped_column(ForeignKey("neurons.id"), index=True)
    miner_id: Mapped[int] = mapped_column(ForeignKey("neurons.id"), index=True)

class EvaluationType(Base):
    __tablename__ = "evaluation_types"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(index=True)

    # Relationship
    solution_scores: Mapped[list["SolutionEvaluation"]] = relationship(back_populates="score_type")

class TaskSolution(Base):
    __tablename__ = "task_solutions"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    challenge_id: Mapped[int] = mapped_column(ForeignKey("challenges.id"), index=True)
    judgement_id: Mapped[int] = mapped_column(ForeignKey("judgements.id"), index=True)
    miner_answer: Mapped[dict] = mapped_column(JSON)

    # Relationship
    challenge: Mapped["Challenge"] = relationship(back_populates="solutions")
    solution_scores: Mapped[list["SolutionEvaluation"]] = relationship(back_populates="solution")

class SolutionEvaluation(Base):
    __tablename__ = "solution_evaluations"
    id: Mapped[int] = mapped_column(primary_key=True)
    solution_id: Mapped[int] = mapped_column(ForeignKey("task_solutions.id"), index=True)
    score_type_id: Mapped[int] = mapped_column(ForeignKey("evaluation_types.id"), index=True)
    value: Mapped[float]

    # Relationships
    score_type: Mapped["EvaluationType"] = relationship(back_populates="solution_scores")
    solution: Mapped["TaskSolution"] = relationship(back_populates="solution_scores")

Base.metadata.create_all(engine)
