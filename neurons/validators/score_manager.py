import bittensor as bt
import copy
import numpy as np

from typing import List

from webgenie.base.neuron import BaseNeuron
from webgenie.constants import CONSIDERING_SESSION_COUNTS
from webgenie.storage import send_challenge_to_stats_collector


class ScoreManager:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron
        self.lock = neuron.lock
        
        self.should_save = False
        
        self.hotkeys = copy.deepcopy(self.neuron.metagraph.hotkeys)
        self.scoring_session = -1
        self.session_accumulated_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        self.last_send_stats_collector_session = -1
        self.winners = []

    def load_scores(self):
        try:
            bt.logging.info("Loading scores")
            state = np.load(self.neuron.config.neuron.full_path + "/state.npz")

            self.hotkeys = state.get(
                "hotkeys",
                copy.deepcopy(self.neuron.metagraph.hotkeys)
            )
            self.scoring_session = state.get(
                "scoring_session", 
                -1
            )
            self.session_accumulated_scores = state.get(
                "tempo_accumulated_scores",
                np.zeros(self.neuron.metagraph.n, dtype=np.float32)
            )
            self.last_send_stats_collector_session = state.get(
                "last_send_stats_collector_session",
                -1
            )
            self.winners = state.get(
                "winners",
                []
            )
        except Exception as e:
            bt.logging.warning(f"Error loading scores: {e}")

    def save_scores(self):
        if not self.should_save:
            return
        
        self.should_save = False
        bt.logging.info("Saving scores")
        np.savez(
            self.neuron.config.neuron.full_path + "/state.npz",
            hotkeys=self.hotkeys,
            scoring_session=self.scoring_session,
            tempo_accumulated_scores=self.session_accumulated_scores,
            last_send_stats_collector_session=self.last_send_stats_collector_session,
            winners=self.winners,
        )
    
    def set_new_hotkeys(self, new_hotkeys: List[str]):
        bt.logging.info(
            "Hotkeys updated, re-syncing scores"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != new_hotkeys[uid]:
                self.session_accumulated_scores[uid] = 0

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(new_hotkeys):
            new_session_accumulated_scores = np.zeros((len(new_hotkeys)))
            min_len = min(len(self.hotkeys), len(self.session_accumulated_scores))
            new_session_accumulated_scores[:min_len] = self.session_accumulated_scores[:min_len]
            self.session_accumulated_scores = new_session_accumulated_scores

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(new_hotkeys)
        self.should_save = True

    def update_scores(self, rewards: np.ndarray, uids: List[int], session: int):
        if self.scoring_session != session:
            # This is a new session, reset the scores and winners.
            self.scoring_session = session
            self.session_accumulated_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
            if len(self.winners) > CONSIDERING_SESSION_COUNTS:
                self.winners.pop(0)
            self.winners.append(-1)

        if not self.winners:
            self.winners.append(-1)

        # Update accumulated scores and track best performer
        self.session_accumulated_scores[uids] += rewards
        bt.logging.info(f"Updated scores: {self.session_accumulated_scores}")
        self.winners[-1] = np.argmax(self.session_accumulated_scores)
        bt.logging.info(f"Updated winners: {self.winners}")
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
        with self.lock:
            scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
            for winner in self.winners:
                scores[winner] += 1

        return scores