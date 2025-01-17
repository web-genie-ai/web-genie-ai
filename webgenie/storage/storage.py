import bittensor as bt

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Miner, Base


engine = create_engine('sqlite:///webgenie.db')
Session = sessionmaker(bind=engine)
Base.metadata.create_all(engine)


def upload_competition(row: dict):
    try:
        session = Session()
        competition = Competition(**row)
        session.add(competition)
        session.commit()
    except Exception as e:
        bt.logging.error(f"Database error: {e}")
        session.rollback()
    finally:
        session.close()


def upload_competition_result(result: dict):
    try:
        session = Session()
        miner = Miner(**result)
        session.add(miner)
        session.commit()
    except Exception as e:
        bt.logging.error(f"Database error: {e}")
        session.rollback()
    finally:
        session.close()
