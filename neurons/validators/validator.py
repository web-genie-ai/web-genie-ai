# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2024 pycorn
import asyncio
from dotenv import load_dotenv
load_dotenv()
import time

import bittensor as bt

from webgenie.base.validator import BaseValidatorNeuron
from webgenie.protocol import WebgenieStreamingSynapse
from neurons.validators.genie_validator import GenieValidator

class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        bt.logging.info("load_state()")
        self.load_state()
        self.genie_validator = GenieValidator()

    async def organic_forward(self, synapse: WebgenieStreamingSynapse):
        return await self.genie_validator.organic_forward(synapse)

    async def forward(self):
        return await self.genie_validator.forward(self)

    async def concurrent_forward(self):
        coroutines = [
            self.forward()
            for _ in range(self.config.neuron.num_concurrent_forwards)
        ]
        await asyncio.gather(*coroutines)

    async def forward_loop(self):
        self.sync()
        bt.logging.info(f"Validator starting at block: {self.block}")

        while True:
            try:
                bt.logging.info(f"step({self.step}) block({self.block})")
                self.loop.run_until_complete(self.concurrent_forward())
                self.sync()
                self.step += 1
            except Exception as e:
                bt.logging.error(f"Error during forward loop: {str(e)}")

            await asyncio.sleep(5)

    async def scoring_loop(self):
        bt.logging.info(f"Scoring loop starting")
        while True:
            try:
                await self.genie_validator.score()
            except Exception as e:
                bt.logging.error(f"Error during scoring: {str(e)}")

            await asyncio.sleep(5)

    async def __aenter__(self):
        self.loop.create_task(self.forward_loop())
        self.loop.create_task(self.scoring_loop())
        self.is_running = True
        bt.logging.debug("Starting validator in background thread")
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.is_running:
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
