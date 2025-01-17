# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 pycorn, Sangar

import bittensor as bt
import asyncio
import copy
import numpy as np
import threading
import time

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename=".env.validator"))
    
from typing import Tuple, Union

from webgenie.base.validator import BaseValidatorNeuron
from webgenie.constants import API_HOTKEY
from webgenie.protocol import WebgenieTextSynapse, WebgenieImageSynapse
from webgenie.utils.uids import get_validator_index

from neurons.validators.genie_validator import GenieValidator
from neurons.validators.score_manager import ScoreManager

# Constants for block timing
BLOCK_IN_SECONDS = 12
TEMPO_BLOCKS = 60
MAX_VALIDATORS = 1
VALIDATOR_QUERY_PERIOD_BLOCKS = 10
ALL_VALIDATOR_QUERY_PERIOD_BLOCKS = MAX_VALIDATORS * VALIDATOR_QUERY_PERIOD_BLOCKS
COMPETITION_PERIOD_BLOCKS = TEMPO_BLOCKS * 3
SET_WEIGHTS_PERIOD_BLOCKS = 50 # 50 blocks = 10 minutes 


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """
    
    @property
    def session_number(self):
        return self.block // COMPETITION_PERIOD_BLOCKS

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
        self.set_weights_thread: Union[threading.Thread, None] = None
        self.lock = threading.Lock()
        
        self.genie_validator = GenieValidator(neuron=self)
        self.score_manager = ScoreManager(neuron=self)

        bt.logging.info("load_state()")
        self.load_state()

        self.sync()

        if not self.config.axon_off:
            self.serve_axon()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        # Copies state of metagraph before syncing.
        previous_metagraph = copy.deepcopy(self.metagraph)

        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)

        # Check if the metagraph axon info has changed.
        if previous_metagraph.axons == self.metagraph.axons:
            return

        bt.logging.info(
            "Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages"
        )
        self.score_manager.set_new_hotkeys(self.metagraph.hotkeys)

    def save_state(self):
        """Saves the state of the validator to a file."""
        self.score_manager.save_scores()
        
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
            bt.logging.info(f"Validator running in organic mode on port {self.config.neuron.axon_port}")
        except Exception as e:
            bt.logging.error(f"Failed to serve Axon with exception: {e}")

    def query_miners_loop(self):    
        bt.logging.info(f"Query miners loop starting")
        while True:
            time.sleep(1)
            try:
                with self.lock:
                    self.sync()
                    validator_index = get_validator_index(self, self.uid)
                    if validator_index == -1:
                        continue
                    validator_index = 0

                # Only allow first N validators to query miners
                # if validator_index > MAX_VALIDATORS:
                #     continue

                # Calculate query period blocks
                with self.lock:
                    current_block = self.block

                start_period_block = (
                    (current_block // ALL_VALIDATOR_QUERY_PERIOD_BLOCKS) * ALL_VALIDATOR_QUERY_PERIOD_BLOCKS + 
                    validator_index * VALIDATOR_QUERY_PERIOD_BLOCKS
                )
                end_period_block = start_period_block + ALL_VALIDATOR_QUERY_PERIOD_BLOCKS / 2
                bt.logging.info(f"Query period blocks - "
                                f"Start: {start_period_block}, "
                                f"End: {end_period_block}, "
                                f"Current: {current_block}")
                # Sleep if outside query window
                if current_block < start_period_block:
                    bt.logging.info(f"Sleeping for {start_period_block - current_block} blocks before querying miners")
                    time.sleep((start_period_block - current_block) * BLOCK_IN_SECONDS)
                elif current_block >= end_period_block:
                    sleep_blocks = (start_period_block - current_block + ALL_VALIDATOR_QUERY_PERIOD_BLOCKS)
                    bt.logging.info(f"Sleeping for {sleep_blocks} blocks before querying miners")
                    time.sleep(sleep_blocks * BLOCK_IN_SECONDS)
                    continue

                self.query_miners_event_loop.run_until_complete(self.genie_validator.query_miners())
            except Exception as e:
                bt.logging.error(f"Error during query miners loop: {str(e)}")
            if self.should_exit:
                break

    def score_loop(self):
        bt.logging.info(f"Scoring loop starting")
        while True:
            time.sleep(1)
            try:
                with self.lock:
                    self.sync()
                self.score_event_loop.run_until_complete(self.genie_validator.score())
            except Exception as e:
                bt.logging.error(f"Error during scoring: {str(e)}")
            if self.should_exit:
                break
            time.sleep(1)

    def synthensize_task_loop(self):
        bt.logging.info(f"Synthensize task loop starting")
        while True:
            time.sleep(1)
            try:
                with self.lock:
                    self.sync()
                
                self.synthensize_task_event_loop.run_until_complete(self.genie_validator.synthensize_task())
            except Exception as e:
                bt.logging.error(f"Error during synthensize task: {str(e)}")
            if self.should_exit:
                break
    
    def set_weights_loop(self):
        """
        Every three tempos, set the weights.
        """
        bt.logging.info(f"Set weights loop starting")
        
        while True:
            time.sleep(1)
            try:
                with self.lock:
                    self.sync()
                    current_block = self.block

                # Calculate the end block number for the next weight setting period
                # This aligns with 3 tempo boundaries
                set_weights_end_block = (
                    (current_block + COMPETITION_PERIOD_BLOCKS - 1) 
                    // COMPETITION_PERIOD_BLOCKS 
                    * COMPETITION_PERIOD_BLOCKS
                )

                # Start setting weights 50 blocks before the end
                set_weights_start_block = set_weights_end_block - SET_WEIGHTS_PERIOD_BLOCKS
                bt.logging.info(
                    f"Set weights blocks - "
                    f"Start: {set_weights_start_block}, "
                    f"End: {set_weights_end_block}, "
                    f"Current: {current_block}"
                )
                # Check if we're in the weight setting window
                if (current_block >= set_weights_start_block and 
                    current_block < set_weights_end_block):
                    
                    bt.logging.info(f"Trying to set weights at block {current_block}")
                    self.score_manager.set_weights()
                else:
                    # Sleep until next weight setting window
                    sleep_blocks = set_weights_start_block - current_block
                    bt.logging.info(f"Sleeping for {sleep_blocks} blocks before setting weights")
                    time.sleep(sleep_blocks * BLOCK_IN_SECONDS)
            except Exception as e:
                bt.logging.error(f"Error during set weights: {str(e)}")
            if self.should_exit:
                break

    def run_background_threads(self):
        if not self.is_running:
            bt.logging.info("Starting validator in background thread")
            self.is_running = True
            self.should_exit = False
            
            self.synthensize_task_thread = threading.Thread(target=self.synthensize_task_loop, daemon=True)
            self.query_miners_thread = threading.Thread(target=self.query_miners_loop, daemon=True)
            self.score_thread = threading.Thread(target=self.score_loop, daemon=True)
            self.set_weights_thread = threading.Thread(target=self.set_weights_loop, daemon=True)

            self.synthensize_task_thread.start()
            self.query_miners_thread.start()
            self.score_thread.start()
            self.set_weights_thread.start()
            bt.logging.info("Started background threads")
    
    def stop_background_threads(self):
        if self.is_running:
            bt.logging.info("Stopping background threads")
            self.should_exit = True
            self.is_running = False
            
            self.synthensize_task_thread.join(5)
            self.query_miners_thread.join(5)
            self.score_thread.join(5)
            self.set_weights_thread.join(5)

            self.synthensize_task_thread = None
            self.query_miners_thread = None
            self.score_thread = None
            self.set_weights_thread = None
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

