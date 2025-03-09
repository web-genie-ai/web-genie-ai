# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 pycorn, Sangar

import bittensor as bt
import asyncio
import copy
import numpy as np
import threading
import time

from dotenv import load_dotenv
load_dotenv(".env.validator")
load_dotenv(".env")

from rich.table import Table
from rich.console import Console
from typing import Tuple, Union

from webgenie.base.validator import BaseValidatorNeuron
from webgenie.base.utils.weight_utils import (
    process_weights_for_netuid,
    convert_weights_and_uids_for_emit,
) 
from webgenie.helpers.ports import kill_process_on_port
from webgenie.constants import (
    API_HOTKEY,
    BLOCK_IN_SECONDS,
    SESSION_WINDOW_BLOCKS,
    QUERING_WINDOW_BLOCKS,
    WEIGHT_SETTING_WINDOW_BLOCKS,
    AXON_OFF,
)
from webgenie.protocol import WebgenieTextSynapse, WebgenieImageSynapse
from webgenie.rewards.lighthouse_reward import start_lighthouse_server_thread, stop_lighthouse_server
from webgenie.storage import send_challenge_to_stats_collector
from webgenie.utils.uids import get_validator_index

from neurons.validators.genie_validator import GenieValidator
from neurons.validators.score_manager import ScoreManager


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """
    
    @property
    def session(self):
        return self.block // SESSION_WINDOW_BLOCKS

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        
        # Create asyncio event loop to manage async tasks.
        self.synthensize_task_event_loop = asyncio.new_event_loop()
        self.query_miners_event_loop = asyncio.new_event_loop()
        self.score_event_loop = asyncio.new_event_loop()
        self.set_weights_event_loop = asyncio.new_event_loop()

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.synthensize_task_thread: Union[threading.Thread, None] = None
        self.query_miners_thread: Union[threading.Thread, None] = None
        self.score_thread: Union[threading.Thread, None] = None
        self.sync_thread: Union[threading.Thread, None] = None
        self.lock = threading.RLock()
        
        self.genie_validator = GenieValidator(neuron=self)
        self.score_manager = ScoreManager(neuron=self)

        bt.logging.info("load_state()")
        self.load_state()

        self.sync()

        if not AXON_OFF:
            self.serve_axon()
            
    def resync_metagraph(self):
        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and miner scores"
        )
        self.score_manager.set_new_hotkeys(self.metagraph.hotkeys)

    def print_weights(self, raw_weights: np.ndarray):
        weights_table = Table(
            title="Raw Weights (Sorted)",
            show_header=True,
            header_style="bold magenta",
            title_style="bold blue", 
            border_style="blue"
        )
        weights_table.add_column("UID", justify="right", style="cyan", header_style="bold cyan")
        weights_table.add_column("Weight", justify="right", style="green")

        # Create list of (uid, weight) tuples and sort by weight descending
        uid_weights = [(uid, weight) for uid, weight in enumerate(raw_weights) if weight > 0]
        uid_weights.sort(key=lambda x: x[1], reverse=True)

        # Add rows to table
        for uid, weight in uid_weights:
            weights_table.add_row(
                str(uid),
                f"{weight:.4f}"
            )

        console = Console()
        console.print(weights_table)

    def set_weights(self):        
        with self.lock:
            current_session = self.session
            last_set_weights_session = self.score_manager.last_set_weights_session
            if last_set_weights_session == current_session - 1:
                return

        scores = self.score_manager.get_scores(current_session - 1)
        if np.all(scores == 0):
            bt.logging.info(f"All scores are 0, skipping set_weights")
            return
        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        # Compute the norm of the scores
        norm = np.linalg.norm(scores, ord=1, axis=0, keepdims=True)

        # Check if the norm is zero or contains NaN values
        if np.any(norm == 0) or np.isnan(norm).any():
            norm = np.ones_like(norm)  # Avoid division by zero or NaN

        # Compute raw_weights safely
        raw_weights = scores / norm

        self.print_weights(raw_weights)

        return
        with self.lock:
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

            # Convert to uint16 weights and uids.
            (
                uint_uids,
                uint_weights,
            ) = convert_weights_and_uids_for_emit(
                uids=processed_weight_uids, weights=processed_weights
            )
            # Set the weights on chain via our subtensor connection.
            result, msg = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uint_uids,
                weights=uint_weights,
                wait_for_finalization=False,
                wait_for_inclusion=False,
                version_key=self.spec_version,
            )
        if result is True:
            self.score_manager.last_set_weights_session = current_session - 1
            with self.lock:
                self.score_manager.save_scores()
                self.score_manager.save_session_result_to_file(current_session-1)
            try:
                bt.logging.info(f"Sending challenge to stats collector for session {current_session-1}")
                send_challenge_to_stats_collector(self.wallet, current_session-1)
            except Exception as e:
                bt.logging.error(f"Error sending challenge to stats collector: {e}")

            bt.logging.success("set_weights on chain successfully!")
        else:
            bt.logging.error("set_weights failed", msg)
        
    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")
        self.score_manager.load_scores()        
    
    async def blacklist_text(self, synapse: WebgenieTextSynapse) -> Tuple[bool, str]:
        """
        Only allow the backend owner to send synapse to the validator.
        """
        if synapse.dendrite.hotkey == API_HOTKEY:
            return False, "Backend hotkey"
        return True, "Blacklisted"  
    
    async def blacklist_image(self, synapse: WebgenieImageSynapse) -> Tuple[bool, str]:
        """
        Only allow the backend owner to send synapse to the validator.
        """
        if synapse.dendrite.hotkey == API_HOTKEY:
            return False, "Backend hotkey"
        return True, "Blacklisted"  
    
    async def organic_forward_text(self, synapse: WebgenieTextSynapse):
        return await self.genie_validator.organic_forward(synapse)

    async def organic_forward_image(self, synapse: WebgenieImageSynapse):
        return await self.genie_validator.organic_forward(synapse)

    def serve_axon(self):
        """Serve axon to enable external connections."""
        bt.logging.info("serving ip to chain...")
        try:
            bt.logging.info(f"Killing process on port {self.config.axon.port}")
            kill_process_on_port(self.config.axon.port)

            self.axon = bt.axon(wallet=self.wallet, config=self.config)
            self.axon.attach(
                forward_fn = self.organic_forward_text,
                blacklist_fn = self.blacklist_text,
            ).attach(
                forward_fn = self.organic_forward_image,
                blacklist_fn = self.blacklist_image,
            )

            self.axon.serve(
                netuid=self.config.netuid,
                subtensor=self.subtensor,
            )
            self.axon.start()
            bt.logging.info(f"Validator running in organic mode on port {self.config.axon.port}")
        except Exception as e:
            bt.logging.error(f"Failed to serve Axon with exception: {e}")

    def query_miners_loop(self):    
        bt.logging.info(f"Query miners loop starting")
        while True:
            time.sleep(1)
            try:
                # validator_index, validator_count = get_validator_index(self, self.uid)
                # if validator_index == -1:
                #     bt.logging.error(f"No enough stake for the validator.")
                #     continue

                # bt.logging.info(f"Validator index: {validator_index}, Validator count: {validator_count}")
                # # Calculate query period blocks
                # with self.lock:
                #     current_block = self.block

                # all_validator_query_period_blocks = validator_count * QUERING_WINDOW_BLOCKS
                # # Calculate query period blocks
                # start_period_block = (
                #     (current_block // all_validator_query_period_blocks) * 
                #     all_validator_query_period_blocks + 
                #     validator_index * QUERING_WINDOW_BLOCKS
                # )
                # end_period_block = start_period_block + QUERING_WINDOW_BLOCKS / 2
                # bt.logging.info(f"Query window - "
                #                 f"Start: {start_period_block}, "
                #                 f"End: {end_period_block}, "
                #                 f"Current: {current_block}")
                # # Sleep if outside query window
                # if current_block < start_period_block:
                #     sleep_blocks = start_period_block - current_block
                #     bt.logging.info(f"Sleeping for {sleep_blocks} blocks before querying miners")
                #     time.sleep(sleep_blocks * BLOCK_IN_SECONDS)
                #     continue
                # elif current_block > end_period_block:
                #     sleep_blocks = (start_period_block - current_block + all_validator_query_period_blocks)
                #     bt.logging.info(f"Sleeping for {sleep_blocks} blocks before querying miners")
                #     time.sleep(sleep_blocks * BLOCK_IN_SECONDS)
                #     continue
                
                # QUERY_MINERS_TIMEOUT = 60 * 15
                # self.query_miners_event_loop.run_until_complete(
                #     asyncio.wait_for(
                #         self.genie_validator.query_miners(),
                #         timeout=QUERY_MINERS_TIMEOUT
                #     )
                # )
                FORWARD_TIMEOUT = 60 * 60 * 2 # 2 hours
                self.query_miners_event_loop.run_until_complete(
                    asyncio.wait_for(
                        self.genie_validator.forward(),
                        timeout=FORWARD_TIMEOUT
                    )
                )
            except Exception as e:
                bt.logging.error(f"Error during query miners loop: {str(e)}")
            if self.should_exit:
                break

    def score_loop(self):
        bt.logging.info(f"Scoring loop starting")
        while True:
            time.sleep(1)
            try:
                SCORE_TIMEOUT = 60 * 60 * 2 # 2 hours
                self.score_event_loop.run_until_complete(
                    asyncio.wait_for(
                        self.genie_validator.score(),
                        timeout=SCORE_TIMEOUT
                    )
                )
            except Exception as e:
                bt.logging.error(f"Error during scoring: {str(e)}")
            if self.should_exit:
                break

    def synthensize_task_loop(self):
        bt.logging.info(f"Synthensize task loop starting")
        while True:
            time.sleep(1)
            try:
                SYNTHETIC_TASK_TIMEOUT = 60 * 15 # 15 minutes
                self.synthensize_task_event_loop.run_until_complete(
                    asyncio.wait_for(
                        self.genie_validator.synthensize_task(),
                        timeout=SYNTHETIC_TASK_TIMEOUT
                    )
                )
            except Exception as e:
                bt.logging.error(f"Error during synthensize task: {str(e)}")
            if self.should_exit:
                break
    
    def sync_loop(self):
        bt.logging.info(f"Sync loop starting")
        
        while True:
            time.sleep(BLOCK_IN_SECONDS * 10)
            try:
                with self.lock:
                    self.sync()
                self.set_weights()
            except Exception as e:
                bt.logging.error(f"Error during sync: {str(e)}")
            if self.should_exit:
                break

    def run_background_threads(self):
        if not self.is_running:
            bt.logging.info("Starting validator in background thread")
            self.is_running = True
            self.should_exit = False
            
            #self.synthensize_task_thread = threading.Thread(target=self.synthensize_task_loop, daemon=True)
            self.query_miners_thread = threading.Thread(target=self.query_miners_loop, daemon=True)
            #self.score_thread = threading.Thread(target=self.score_loop, daemon=True)
            self.sync_thread = threading.Thread(target=self.sync_loop, daemon=True)

            #self.synthensize_task_thread.start()
            self.query_miners_thread.start()
            #self.score_thread.start()
            self.sync_thread.start()        
            start_lighthouse_server_thread()
            bt.logging.info("Started background threads")
            bt.logging.info("=" * 40)
    
    def stop_background_threads(self):
        if self.is_running:
            bt.logging.info("Stopping background threads")
            self.should_exit = True
            self.is_running = False
            
            #self.synthensize_task_thread.join(5)
            self.query_miners_thread.join(5)
            #self.score_thread.join(5)
            self.sync_thread.join(5)
            stop_lighthouse_server()

            #self.synthensize_task_thread = None
            self.query_miners_thread = None
            #self.score_thread = None
            self.sync_thread = None
            bt.logging.info("Stopped background threads")

    def __enter__(self):
        self.run_background_threads()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_background_threads()
        

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                bt.logging.info("Keyboard interrupt detected, stopping main loop")
                break

