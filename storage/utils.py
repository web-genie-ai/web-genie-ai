from database import Session as DBSession
from models import Neuron, LeaderboardSession, Competition, Challenge, Judgement, EvaluationType, TaskSolution, SolutionEvaluation
from datetime import datetime, timedelta
import logging
from sqlalchemy import and_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# session period tempos
SESSION_PERIOD = 2

# Create a new session
session = DBSession()

def create_record(session: Session, model_class, **kwargs):
    try:
        new_record = model_class(**kwargs)  # Create an instance of the model
        session.add(new_record)  # Add it to the session
        session.commit()  # Commit the session
        return new_record.id  # Return the new record's ID
    except Exception as e:
        session.rollback()  # Rollback in case of error
        logging.error(f"An error occurred: {e}")
        return None  # Return None to indicate failure
    finally:
        session.close()  # Close the session

def add_neuron(coldkey: str, hotkey: str):
    return create_record(session, Neuron, coldkey=coldkey, hotkey=hotkey)

def get_neuron_id(hotkey: str):
    try:
        neuron = session.query(Neuron).filter_by(hotkey=hotkey).first()
        if neuron:
            return neuron.id
        else:
            return None  # Return None if no matching neuron is found
    except SQLAlchemyError as e:
        logging.error(f"An error occurred while fetching neuron: {e}")
        return None
    finally:
        session.close()  # Ensure the session is closed

def create_leaderboard_session(created_at: datetime, competition_id: int):
    return create_record(session, LeaderboardSession, created_at=created_at, competition_id=competition_id)

def query_leaderboard_session(timestamp: datetime):
    # Calculate the time range
    interval = SESSION_PERIOD * 72 * 60
    try:
        # Query the LeaderboardSession where created_at is within the specified range
        leaderboard_session = session.query(LeaderboardSession).filter(
            and_(
                LeaderboardSession.created_at <= timestamp,
                LeaderboardSession.created_at + timedelta(seconds=interval) > timestamp
            )
        ).first()  # Get the first matching session

        # Return the session_id if found, otherwise None
        return leaderboard_session.id if leaderboard_session else None
    
    except Exception as e:
        logging.error(f"An error occurred while querying leaderboard session: {e}")
        return None  # Return None in case of error
    finally:
        session.close()  # Close the session 

def create_competition(name: str):
    return create_record(session, Competition, name=name)

def create_challenge(session_id: int, ground_truth_html: str):
    return create_record(session, Challenge, session_id=session_id, ground_truth_html=ground_truth_html)

def create_judgement(validator_id: int, miner_id: int):
    return create_record(session, Judgement, validator_id=validator_id, miner_id=miner_id)

def create_evaluation_type(name: str):
    return create_record(session, EvaluationType, name=name)

def create_task_solution(miner_answer: str, challenge_id: int, created_at: datetime):
    return create_record(session, TaskSolution, miner_answer=miner_answer, challenge_id=challenge_id, created_at=created_at)

def create_solution_evaluation(solution_id: int, score_type_id: int, judgement_id: int, value: float):
    return create_record(session, SolutionEvaluation, solution_id=solution_id, score_type_id=score_type_id, judgement_id=judgement_id, value=value)

if __name__ == "__main__":
    neuron_id = add_neuron("5GKH9FPPnWSUoeeTJp19wVtd84XqFW4pyK2ijV2GsFbhTrP1", "5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3")
    logging.info(f"neuron_id: {neuron_id}")

    html = "<div>test</div>"

    create_competition("Accuracy")
    create_competition("SEO")
    create_competition("CODE_QUALITY")
    create_competition("WEIGHTED_SCORE")
    session_id = create_leaderboard_session(datetime.now(), 1)
    challenge = create_challenge(session_id, html)
