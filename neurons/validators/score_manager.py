import bittensor as bt
import copy
import numpy as np
import pickle

from typing import List

from webgenie.base.neuron import BaseNeuron
from webgenie.challenges.challenge import Challenge, RESERVED_WEIGHTS
from webgenie.constants import CONSIDERING_SESSION_COUNTS
from webgenie.storage import send_challenge_to_stats_collector


class ScoreManager:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron
        self.state_path = self.neuron.config.neuron.full_path + "/state.npz"
        self.lock = neuron.lock
        
        self.should_save = False

        self.hotkeys = copy.deepcopy(self.neuron.metagraph.hotkeys)
        self.current_session = -1
        self.total_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        self.last_send_stats_collector_session = -1
        self.winners = {}

    def load_scores(self):
        try:
            bt.logging.info(f"Loading scores from {self.state_path}")
            data = np.load(self.state_path, allow_pickle=True)

            self.hotkeys = data["hotkeys"]
            self.current_session = data["current_session"]
            self.total_scores = data["total_scores"]
            self.last_send_stats_collector_session = data["last_send_stats_collector_session"]
            self.winners = dict(data["winners"].item())
            bt.logging.info(f"Winners: {self.winners}")
        except Exception as e:
            bt.logging.error(f"Error loading state: {e}")
            self.hotkeys = copy.deepcopy(self.neuron.metagraph.hotkeys)
            self.current_session = -1
            self.total_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
            self.last_send_stats_collector_session = -1
            self.winners = {}

    def save_scores(self):
        if not self.should_save:
            return

        try:
            bt.logging.info(f"Saving scores to {self.state_path}")
            np.savez(
                self.state_path, 
                hotkeys=self.hotkeys, 
                current_session=self.current_session, 
                total_scores=self.total_scores, 
                last_send_stats_collector_session=self.last_send_stats_collector_session, 
                winners=self.winners,
                allow_pickle=True,
            )
            self.should_save = False
        except Exception as e:
            bt.logging.error(f"Error saving state: {e}")
    
    def set_new_hotkeys(self, new_hotkeys: List[str]):
        bt.logging.info(
            "Hotkeys updated, re-syncing scores"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != new_hotkeys[uid]:
                self.total_scores[uid] = 0

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(new_hotkeys):
            new_total_scores = np.zeros((len(new_hotkeys)))
            min_len = min(len(self.hotkeys), len(self.total_scores))
            new_total_scores[:min_len] = self.total_scores[:min_len]
            self.total_scores = new_total_scores

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(new_hotkeys)
        self.should_save = True

    def update_scores(self, rewards: np.ndarray, uids: List[int], challenge: Challenge):
        bt.logging.info("Updating scores")
        session = challenge.session
        competition_type = challenge.competition_type
        if self.current_session != session:
            # This is a new session, reset the scores and winners.
            self.current_session = session
            self.total_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        # Update accumulated scores and track best performer
        bt.logging.info(f"Updating scores for uids: {uids}")
        bt.logging.info(f"Rewards: {rewards}")
        bt.logging.info(f"Total scores: {self.total_scores}")
        self.total_scores[uids] += rewards
        bt.logging.info("Updating winners table")
        print(np.argmax(self.total_scores), competition_type)
        print(self.winners)
        self.winners[session] = (np.argmax(self.total_scores), competition_type)
        bt.logging.info(f"Winners: {self.winners}")
        for session_number in self.winners:
            if session_number < session - CONSIDERING_SESSION_COUNTS:
                self.winners.pop(session_number)
        bt.logging.info(f"Winners: {self.winners}")
        self.should_save = True

    def send_challenge_to_stats_collector(self):
        with self.lock:
            current_session = self.neuron.session

        if current_session != self.last_send_stats_collector_session:
            try:
                send_challenge_to_stats_collector(self.neuron.wallet, current_session)
                self.last_send_stats_collector_session = current_session
            except Exception as e:
                bt.logging.error(f"Error sending challenge to stats collector: {e}")
        
    def get_scores(self):
        scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        with self.lock:
            for session_number in self.winners:
                winner, competition_type = self.winners[session_number]
                scores[winner] += RESERVED_WEIGHTS[competition_type]

        return scores