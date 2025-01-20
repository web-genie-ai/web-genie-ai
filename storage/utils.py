from database import Session as DBSession
from models import Neuron, LeaderboardSession, Competition, Challenge, Judgement, EvaluationType, TaskSolution, SolutionEvaluation
from datetime import datetime
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
    # Check if the session with the given id already exists
    existing_neuron = session.query(Neuron).filter_by(hotkey=hotkey).first()
    
    if existing_neuron:
        logging.info(f"neuron with hotkey {hotkey} already exists. Skipping creation.")
        return existing_neuron.id  # Return the existing session id

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

def create_leaderboard_session(session_number: int, created_at: datetime, competition_id: int):
    # Check if the session with the given id already exists
    existing_session = session.query(LeaderboardSession).filter_by(id=session_number).first()
    
    if existing_session:
        logging.info(f"Session with id {session_number} already exists. Skipping creation.")
        return existing_session.id  # Return the existing session id

    return create_record(session, LeaderboardSession,
                         id=session_number,
                         created_at=created_at, competition_id=competition_id)

def create_competition(name: str):
    # Check if the competition with the given name already exists
    existing_competition = session.query(Competition).filter_by(name=name).first()
    if existing_competition:
        logging.info(f"Competition with name {name} already exists. Skipping creation.")
        return existing_competition.id  # Return the existing competition id

    return create_record(session, Competition, name=name)

def create_challenge(session_id: int, ground_truth_html: str):
    return create_record(session, Challenge, session_id=session_id, ground_truth_html=ground_truth_html)

def create_judgement(validator_id: int, miner_id: int):
    return create_record(session, Judgement, validator_id=validator_id, miner_id=miner_id)

def create_evaluation_type(name: str):
    # Check if the competition with the given name already exists
    existing_evaluation_type = session.query(EvaluationType).filter_by(name=name).first()
    if existing_evaluation_type:
        logging.info(f"Evaluation type with name {name} already exists. Skipping creation.")
        return existing_evaluation_type.id  # Return the existing evaluation type id

    return create_record(session, EvaluationType, name=name)

def create_task_solution(miner_answer: dict, challenge_id: int):
    return create_record(session, TaskSolution, miner_answer=miner_answer, challenge_id=challenge_id)

def create_solution_evaluation(solution_id: int, score_type_id: int, judgement_id: int, value: float):
    return create_record(session, SolutionEvaluation, solution_id=solution_id, score_type_id=score_type_id, judgement_id=judgement_id, value=value)

def store_results_to_database(results: dict):
    neuron = results["neuron"]
    vali_coldkey, vali_hotkey = results["validator"]
    miner_uids = results["miner_uids"]
    solutions = results["solutions"]
    scores = results["scores"]
    challenge = results["challenge"]
    session_number = challenge["session_number"]
    block_start_datetime = results["block_start_datetime"]
    ground_truth_html = challenge["task"]
    competition_type = challenge["competition_type"]
    competition_id = create_competition(competition_type)
    session_id = create_leaderboard_session(session_number, block_start_datetime, competition_id)
    challenge_id = create_challenge(session_id, ground_truth_html)
    
    # Iterate over miner_uids to store TaskSolution data
    for miner_uid, solution, score in zip(miner_uids, solutions, scores):
        coldkey = miner_uid["coldkey"]
        hotkey = miner_uid["hotkey"]
        miner_answer = solution['miner_answer']
        neuron_validator_id = add_neuron(vali_coldkey, vali_hotkey)
        neuron_miner_id = add_neuron(coldkey, hotkey)
        judgement_id = create_judgement(neuron_validator_id, neuron_miner_id)
        solution_id = create_task_solution(miner_answer, challenge_id)
        # Collect evaluation types and their scores
        for eval_type, score_value in score.items():
            evaluation_type_id = create_evaluation_type(eval_type)
            create_solution_evaluation(solution_id, evaluation_type_id, judgement_id, score_value)

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
