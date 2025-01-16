import bittensor as bt
import copy
import numpy as np

from typing import List


from webgenie.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
) 
from webgenie.base.neuron import BaseNeuron

class ScoreManager:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron
        self.session_number = 0
        self.hotkeys = copy.deepcopy(self.neuron.metagraph.hotkeys)
        self.scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        self.tempo_accumulated_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)

    def load_scores(self):
        bt.logging.info("Loading scores")
        state = np.load(self.neuron.config.neuron.full_path + "/state.npz")
        try:
            self.scores = state["scores"]
            self.hotkeys = state["hotkeys"]
            self.session_number = state["session_number"]
            self.tempo_accumulated_scores = state["tempo_accumulated_scores"]
        except Exception as e:
            bt.logging.error(f"Error loading scores: {e}")
            self.scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
            self.hotkeys = copy.deepcopy(self.neuron.metagraph.hotkeys)
            self.session_number = 0
            self.tempo_accumulated_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)

    def save_scores(self):
        bt.logging.info("Saving scores")
        np.savez(
            self.neuron.config.neuron.full_path + "/state.npz",
            scores=self.scores,
            hotkeys=self.hotkeys,
            session_number=self.session_number,
            tempo_accumulated_scores=self.tempo_accumulated_scores,
        )
    
    def set_new_hotkeys(self, new_hotkeys: List[str]):
        bt.logging.info(
            "Hotkeys updated, re-syncing scores"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != new_hotkeys[uid]:
                self.tempo_accumulated_scores[uid] = 0
                self.scores[uid] = 0  # hotkey has been replaced

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(new_hotkeys):
            new_scores = np.zeros((len(new_hotkeys)))
            min_len = min(len(self.hotkeys), len(self.scores))
            new_scores[:min_len] = self.scores[:min_len]
            self.scores = new_scores

            new_tempo_accumulated_scores = np.zeros((len(new_hotkeys)))
            min_len = min(len(self.hotkeys), len(self.tempo_accumulated_scores))
            new_tempo_accumulated_scores[:min_len] = self.tempo_accumulated_scores[:min_len]
            self.tempo_accumulated_scores = new_tempo_accumulated_scores

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(new_hotkeys)

    def update_scores(self, rewards: np.ndarray, uids: List[int], session_number: int):
        if self.scoring_session_number != session_number:
            # In the new session, reset the scores
            self.scoring_session_number = session_number
            self.tempo_accumulated_scores = np.zeros_like(self.scores)

        # Check if rewards contains NaN values.
        if np.isnan(rewards).any():
            bt.logging.warning(f"NaN values detected in rewards: {rewards}")
            rewards = np.nan_to_num(rewards, nan=0)

        # Ensure rewards is a numpy array.
        rewards = np.asarray(rewards)

        # Check if `uids` is already a numpy array and copy it to avoid the warning.
        if isinstance(uids, np.ndarray):
            uids_array = uids.copy()
        else:
            uids_array = np.array(uids)

        # Handle edge case: If either rewards or uids_array is empty.
        if rewards.size == 0 or uids_array.size == 0:
            bt.logging.info(f"rewards: {rewards}, uids_array: {uids_array}")
            bt.logging.warning(
                "Either rewards or uids_array is empty. No updates will be performed."
            )
            return

        # Check if sizes of rewards and uids_array match.
        if rewards.size != uids_array.size:
            raise ValueError(
                f"Shape mismatch: rewards array of shape {rewards.shape} "
                f"cannot be broadcast to uids array of shape {uids_array.shape}"
            )

        # Compute forward pass rewards, assumes uids are mutually exclusive.
        # shape: [ metagraph.n ]
        scattered_rewards: np.ndarray = np.zeros_like(self.scores)
        scattered_rewards[uids_array] = rewards
        bt.logging.debug(f"Scattered rewards: {scattered_rewards}")

        self.tempo_accumulated_scores: np.ndarray = scattered_rewards + self.tempo_accumulated_scores
        bt.logging.debug(f"Updated scores: {self.tempo_accumulated_scores}")
    
    def set_weights(self):
        """
        Sets the validator weights to the metagraph hotkeys based on the scores it has received from the miners. The weights determine the trust and incentive level the validator assigns to miner nodes on the network.
        """
        self.scores = np.zeros_like(self.scores)
        best_index = np.argmax(self.tempo_accumulated_scores)
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

        bt.logging.debug("raw_weights", raw_weights)
        bt.logging.debug("raw_weight_uids", str(self.metagraph.uids.tolist()))
        # Process the raw weights to final_weights via subtensor limitations.
        (
            processed_weight_uids,
            processed_weights,
        ) = process_weights_for_netuid(
            uids=self.metagraph.uids,
            weights=raw_weights,
            netuid=self.config.netuid,
            subtensor=self.subtensor,
            metagraph=self.metagraph,
        )
        bt.logging.debug("processed_weights", processed_weights)
        bt.logging.debug("processed_weight_uids", processed_weight_uids)

        # Convert to uint16 weights and uids.
        (
            uint_uids,
            uint_weights,
        ) = convert_weights_and_uids_for_emit(
            uids=processed_weight_uids, weights=processed_weights
        )
        bt.logging.debug("uint_weights", uint_weights)
        bt.logging.debug("uint_uids", uint_uids)

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
            bt.logging.info("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)
            