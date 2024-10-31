# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import asyncio
import bittensor as bt

from typing import Awaitable, List
from dataclasses import dataclass
from btcopilot.protocol import BtCopilotSynapse
from btcopilot.validator.reward import get_rewards
from btcopilot.utils.uids import get_random_uids
from btcopilot.solution import Solution

async def process_response(uid: int, async_generator: Awaitable):
    try:
        buffer = ""
        chunk = None
        async for chunk in async_generator:
            if isinstance(chunk, str):
                buffer += chunk
        if chunk is not None:
            synapse = chunk
            if isinstance(synapse, BtCopilotSynapse):
                if synapse.dendrite.status_code == 200:
                    synapse.solution.miner_uid = uid
                    return synapse.solution
            else:
                bt.logging.error(f"Received non-200 status code: {chunk.dendrite.status_code} for uid: {uid}")
                return None
        else:
            bt.logging.error(f"Synapse is None for uid: {uid}")
            return None
    except Exception as e:
        bt.logging.error(f"Error processing response for uid: {uid}: {e}")
        return None

async def handle_responses(miner_uids_list: List[int], responses: List[Awaitable]):
    tasks = [process_response(uid, response) for uid, response in zip(miner_uids_list, responses)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [result for result in results if result is not None]

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.
    miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
    miner_uids_list = miner_uids.tolist()

    bt.logging.info(f"Selected miners: {miner_uids}")

    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    
    task = self.task_generator.next_task()

    synapse = BtCopilotSynapse(
        task=task
    )

    # The dendrite client queries the network.
    responses = await self.dendrite(
        axons=axons,
        synapse=synapse,
        timeout=task.timeout,
        deserialize=True,
        streaming=True,
    )

    handle_responses_task = asyncio.create_task(handle_responses(miner_uids_list, responses))
    
    results = await handle_responses_task
    if len(results) == 0:
        bt.logging.info("No responses received")
        return
    bt.logging.info(f"Received {results} results")

    scores, miner_uids = self.reward_manager.score(task, results)
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    bt.logging.info(f"Updating scores: {scores}")
    self.update_scores(scores, miner_uids)
    time.sleep(5)