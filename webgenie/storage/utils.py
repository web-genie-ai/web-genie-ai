import bittensor as bt
from io import BufferedReader
from .database import Session as DBSession
from .models import Neuron, LeaderboardSession, Competition, Challenge, Judgement, EvaluationType, TaskSolution, SolutionEvaluation
from datetime import datetime
from sqlalchemy import and_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
import json
import time
import requests

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
        bt.logging.error(f"An error occurred: {e}")
        return None  # Return None to indicate failure
    finally:
        session.close()  # Close the session

def add_neuron(coldkey: str, hotkey: str):
    # Check if the session with the given id already exists
    existing_neuron = session.query(Neuron).filter_by(hotkey=hotkey).first()
    
    if existing_neuron:
        bt.logging.info(f"neuron with hotkey {hotkey} already exists. Skipping creation.")
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
        bt.logging.error(f"An error occurred while fetching neuron: {e}")
        return None
    finally:
        session.close()  # Ensure the session is closed

def create_leaderboard_session(session_number: int, created_at: datetime, competition_id: int):
    # Check if the session with the given id already exists
    existing_session = session.query(LeaderboardSession).filter_by(id=session_number).first()
    
    if existing_session:
        bt.logging.info(f"Session with id {session_number} already exists. Skipping creation.")
        return existing_session.id  # Return the existing session id

    return create_record(session, LeaderboardSession,
                         id=session_number,
                         created_at=created_at, competition_id=competition_id)

def create_competition(name: str):
    # Check if the competition with the given name already exists
    existing_competition = session.query(Competition).filter_by(name=name).first()
    if existing_competition:
        bt.logging.info(f"Competition with name {name} already exists. Skipping creation.")
        return existing_competition.id  # Return the existing competition id

    return create_record(session, Competition, name=name)

def create_challenge(session_id: int, ground_truth_html: str):
    return create_record(session, Challenge, session_id=session_id, ground_truth_html=ground_truth_html)

def create_judgement(validator_id: int, miner_id: int):
    # Check if the judgement with the given validator and miner id already exists
    existing_judgement = session.query(Judgement).filter_by(validator_id=validator_id, miner_id=miner_id).first()
    if existing_judgement:
        bt.logging.info(f"judgement with given {validator_id} and {miner_id} already exists. Skipping creation.")
        return existing_judgement.id  # Return the existing competition id

    return create_record(session, Judgement, validator_id=validator_id, miner_id=miner_id)

def create_evaluation_type(name: str):
    # Check if the competition with the given name already exists
    existing_evaluation_type = session.query(EvaluationType).filter_by(name=name).first()
    if existing_evaluation_type:
        bt.logging.info(f"Evaluation type with name {name} already exists. Skipping creation.")
        return existing_evaluation_type.id  # Return the existing evaluation type id

    return create_record(session, EvaluationType, name=name)

def create_task_solution(miner_answer: dict, challenge_id: int):
    return create_record(session, TaskSolution, miner_answer=miner_answer, challenge_id=challenge_id)

def create_solution_evaluation(solution_id: int, score_type_id: int, judgement_id: int, value: float):
    return create_record(session, SolutionEvaluation, solution_id=solution_id, score_type_id=score_type_id, judgement_id=judgement_id, value=value)

def store_results_to_database(results: dict):
    # Extracting validator keys correctly
    vali_coldkey = results["validator"]["coldkey"]
    vali_hotkey = results["validator"]["hotkey"]
    
    # Extracting miners, solutions, scores, and challenge details
    miners = results["miners"]
    solutions = results["solutions"]
    scores = results["scores"]
    challenge = results["challenge"]

    session_number = challenge["session_number"]
    session_start_datetime = results["session_start_datetime"]
    ground_truth_html = challenge["task"]
    competition_type = challenge["competition_type"]

    competition_id = create_competition(competition_type)
    session_id = create_leaderboard_session(session_number, session_start_datetime, competition_id)
    challenge_id = create_challenge(session_id, ground_truth_html)
    
    # Iterate over miner_uids to store TaskSolution data
    for miner, solution, score in zip(miners, solutions, scores):
        coldkey = miner["coldkey"]
        hotkey = miner["hotkey"]
        miner_answer = solution['miner_answer']
        neuron_validator_id = add_neuron(vali_coldkey, vali_hotkey)
        neuron_miner_id = add_neuron(coldkey, hotkey)
        judgement_id = create_judgement(neuron_validator_id, neuron_miner_id)
        solution_id = create_task_solution(miner_answer, challenge_id)
        # Collect evaluation types and their scores
        for eval_type, score_value in score.items():
            evaluation_type_id = create_evaluation_type(eval_type)
            create_solution_evaluation(solution_id, evaluation_type_id, judgement_id, score_value)

def get_session_data(session_number: int):
    try:
        competition = session.query(Competition).join(LeaderboardSession).filter(
            LeaderboardSession.id == session_number
        ).first()

        if not competition:
            return {}
        
        # Constructing the payload
        payload = {
            "external_id": competition.id,
            "name": competition.name,
            "leaderboard_sessions": []
        }

        for leaderboard_session in competition.sessions:
            if leaderboard_session.id != session_number:
                continue  # Skip sessions that do not match the session number
            
            session_data = {
                "external_id": leaderboard_session.id,
                "created_at": leaderboard_session.created_at.isoformat(),
                "challenges": []
            }

            for challenge in leaderboard_session.challenges:
                challenge_data = {
                    "external_id": challenge.id,
                    "ground_truth_html": challenge.ground_truth_html,
                    "task_solutions": []
                }

                for task_solution in challenge.solutions:
                    solution_data = {
                        "external_id": task_solution.id,
                        "created_at": task_solution.created_at.isoformat(),
                        "miner_answer": task_solution.miner_answer,
                        "solution_evaluations": []
                    }

                    # Retrieve evaluations for the task solution
                    for evaluation in task_solution.solution_scores:
                        judgement = session.query(Judgement).get(evaluation.judgement_id)
                        miner_neuron = session.query(Neuron).get(judgement.miner_id)
                        validator_neuron = session.query(Neuron).get(judgement.validator_id)

                        evaluation_data = {
                            "external_id": evaluation.id,
                            "judgement": {
                                "external_id": judgement.id,
                                "miner": miner_neuron.hotkey,  # Assuming hotkey is used as identifier
                                "validator": validator_neuron.hotkey
                            },
                            "evaluation_type": {
                                "external_id": evaluation.score_type_id,
                                "name": session.query(EvaluationType).get(evaluation.score_type_id).name
                            },
                            "value": evaluation.value
                        }

                        solution_data["solution_evaluations"].append(evaluation_data)

                    challenge_data["task_solutions"].append(solution_data)

                session_data["challenges"].append(challenge_data)

            payload["leaderboard_sessions"].append(session_data)

        return payload
    except SQLAlchemyError as e:
        bt.logging.error(f"An error occurred while fetching neuron: {e}")
        return None
    finally:
        session.close()  

def make_signed_request(
    wallet: "bt.Wallet",
    url: str,
    subnet_id: int,
    payload: dict,
    method: str = 'POST',
    file_path: str | None = None,
    subnet_chain: str = 'mainnet',
) -> requests.Response:
    headers = {
        'Realm': subnet_chain,
        'SubnetID': str(subnet_id),
        'Nonce': str(time.time()),
        'Hotkey': wallet.hotkey.ss58_address,
    }

    file_content = b""
    files = None
    if file_path:
        # TODO: start context for opening file
        opened_file = open(file_path, "rb")
        files = {"file": opened_file}
        file = files.get("file")

        if isinstance(file, BufferedReader):
            file_content = file.read()
            file.seek(0)

    headers_str = json.dumps(headers, sort_keys=True)
    data_to_sign = f"{method}{url}{headers_str}{file_content.decode(errors='ignore')}".encode()
    signature = wallet.hotkey.sign(
        data_to_sign,
    ).hex()
    headers["Signature"] = signature

    response = requests.request(method, url, headers=headers, files=files, json=payload, timeout=5)
    return response

def send_challenge_to_stats_collector(wallet: "bt.Wallet", session_number: int) -> None:
    session_data = get_session_data(session_number)
    response = make_signed_request(
        wallet=wallet,
        url="https://webgenie-collector.bactensor.io/api/competitions/",
        subnet_id=54,
        method="POST",
        payload=session_data,
    )
    if not response.ok:
        print(response.json())