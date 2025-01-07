# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 pycorn, Sangar

import bittensor as bt
import asyncio

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(filename=".env.validator"))
    
from typing import Tuple

from webgenie.base.validator import BaseValidatorNeuron
from webgenie.constants import API_HOTKEY
from webgenie.protocol import WebgenieTextSynapse, WebgenieImageSynapse

from neurons.validators.genie_validator import GenieValidator


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        if not self.config.axon_off:
            self.serve_axon()
        
        self.genie_validator = GenieValidator(neuron=self)

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
            pass

    async def query_miners(self):
        return await self.genie_validator.query_miners()

    async def concurrent_query(self):
        coroutines = [
            self.query_miners()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    async def query_miners_loop(self):
        bt.logging.info(f"Validator starting at block: {self.block}")
        self.sync()
        while True:
            try:
                self.loop.run_until_complete(self.concurrent_query())
                self.sync()
            except Exception as e:
                bt.logging.error(f"Error during forward loop: {str(e)}")
            await asyncio.sleep(1)

    async def score_loop(self):
        bt.logging.info(f"Scoring loop starting")
        while True:
            try:
                await self.genie_validator.score()
                self.sync()
            except Exception as e:
                bt.logging.error(f"Error during scoring: {str(e)}")
            await asyncio.sleep(1)

    async def synthensize_task_loop(self):
        bt.logging.info(f"Synthensize task loop starting")
        while True:
            try:
                await self.genie_validator.synthensize_task()
            except Exception as e:
                bt.logging.error(f"Error during synthensize task: {str(e)}")
            await asyncio.sleep(1)

    async def __aenter__(self):
        self.loop.create_task(self.synthensize_task_loop())
        self.loop.create_task(self.query_miners_loop())
        self.loop.create_task(self.score_loop())
        self.is_running = True

        bt.logging.debug("Starting validator in background thread")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if not self.is_running:
            return
        
        self.should_exit = True
        self.is_running = False
        bt.logging.debug("Stopping validator in background thread")

async def main():
    async with Validator() as validator:
        while validator.is_running and not validator.should_exit:
            await asyncio.sleep(15)    
    
# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    asyncio.run(main())
