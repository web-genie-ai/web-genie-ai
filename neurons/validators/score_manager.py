import bittensor as bt
import copy
import numpy as np

from typing import List


from webgenie.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
) 
from webgenie.base.neuron import BaseNeuron

from webgenie.storage import send_challenge_to_stats_collector


class ScoreManager:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron
        self.scoring_session_number = 0
        self.hotkeys = copy.deepcopy(self.neuron.metagraph.hotkeys)
        self.scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        self.session_accumulated_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        self.should_save = False
        self.last_send_stats_collector_session_number = -1

    def load_scores(self):
        try:
            bt.logging.info("Loading scores")
            state = np.load(self.neuron.config.neuron.full_path + "/state.npz")
            self.scores = state["scores"]
            self.hotkeys = state["hotkeys"]
            self.scoring_session_number = state["scoring_session_number"]
            self.session_accumulated_scores = state["tempo_accumulated_scores"]
        except Exception as e:
            bt.logging.warning(f"Error loading scores: {e}")
            self.scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
            self.hotkeys = copy.deepcopy(self.neuron.metagraph.hotkeys)
            self.scoring_session_number = 0
            self.session_accumulated_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)

    def save_scores(self):
        if not self.should_save:
            return
        
        self.should_save = False
        bt.logging.info("Saving scores")
        np.savez(
            self.neuron.config.neuron.full_path + "/state.npz",
            scores=self.scores,
            hotkeys=self.hotkeys,
            scoring_session_number=self.scoring_session_number,
            tempo_accumulated_scores=self.session_accumulated_scores,
        )
    
    def set_new_hotkeys(self, new_hotkeys: List[str]):
        bt.logging.info(
            "Hotkeys updated, re-syncing scores"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != new_hotkeys[uid]:
                self.session_accumulated_scores[uid] = 0
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(new_hotkeys):
            new_scores = np.zeros((len(new_hotkeys)))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_scores[:min_len] = self.scores[:min_len]
            self.scores = new_scores

            new_tempo_accumulated_scores = np.zeros((len(new_hotkeys)))
            min_len = min(len(self.hotkeys), len(self.session_accumulated_scores))
            new_tempo_accumulated_scores[:min_len] = self.session_accumulated_scores[:min_len]
            self.session_accumulated_scores = new_tempo_accumulated_scores

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(new_hotkeys)
        self.should_save = True

    def update_scores(self, rewards: np.ndarray, uids: List[int], session_number: int):
        if self.scoring_session_number != session_number:
            self.scoring_session_number = session_number
            self.session_accumulated_scores = np.zeros_like(self.scores)

        scattered_rewards: np.ndarray = np.zeros_like(self.scores)
        scattered_rewards[uids] = rewards
        self.session_accumulated_scores: np.ndarray = scattered_rewards + self.session_accumulated_scores
        bt.logging.debug(f"Updated scores: {self.session_accumulated_scores}")
        
        self.should_save = True
    
    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """
        if not self.neuron.should_set_weights():
            return
        
        with self.neuron.lock:
            current_session_number = self.neuron.session_number
            
        if current_session_number != self.last_send_stats_collector_session_number:
            send_challenge_to_stats_collector(self.neuron.wallet, current_session_number)
            self.last_send_stats_collector_session_number = current_session_number

        self.scores = np.zeros_like(self.scores)
        best_index = np.argmax(self.session_accumulated_scores)
        self.scores[best_index] = 1

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        norm = np.linalg.norm(self.scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = self.scores / norm
        
        with self.neuron.lock:
            # Process the raw weights to final_weights via subtensor limitations.
            (
                processed_weight_uids,
                processed_weights,
            ) = process_weights_for_netuid(
                uids=self.neuron.metagraph.uids,
                weights=raw_weights,
                netuid=self.neuron.config.netuid,
                subtensor=self.neuron.subtensor,
                metagraph=self.neuron.metagraph,
            )

            # Convert to uint16 weights and uids.
            (
                uint_uids,
                uint_weights,
            ) = convert_weights_and_uids_for_emit(
                uids=processed_weight_uids, weights=processed_weights
            )
            # Set the weights on chain via our subtensor connection.
            result, msg = self.neuron.subtensor.set_weights(
                wallet=self.neuron.wallet,
                netuid=self.neuron.config.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_finalization=False,
                wait_for_inclusion=False,
                version_key=self.neuron.spec_version,
            )
            if result is True:
                bt.logging.success("set_weights on chain successfully!")
            else:
                bt.logging.error("set_weights failed", msg)
                