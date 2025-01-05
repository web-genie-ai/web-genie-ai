import os
import bittensor as bt
import numpy as np
import random
from typing import Union, List

from webgenie.base.neuron import BaseNeuron
from webgenie.constants import (
    MAX_SYNTHETIC_HISTORY_SIZE, 
    MAX_SYNTHETIC_TASK_SIZE, 
    MAX_DEBUG_IMAGE_STRING_LENGTH, 
    MIN_SYNTHETIC_HISTORY_SIZE_TO_SCORE,
    UPDATE_SCORE_STEPS,
    WORK_DIR
)
from webgenie.helpers.htmls import preprocess_html
from webgenie.protocol import WebgenieImageSynapse, WebgenieTextSynapse
from webgenie.tasks.solution import Solution
from webgenie.tasks.image_task_generator import ImageTaskGenerator
from webgenie.tasks.text_task_generator import TextTaskGenerator
from webgenie.utils.uids import get_random_uids

class GenieValidator:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron
        self.config = neuron.config
        self.synthetic_history = []
        self.synthetic_tasks = []
        self.un_responsed_count = np.zeros(self.neuron.metagraph.n, dtype=np.float32)

        self.task_generators = [
            (TextTaskGenerator(), 0.1),
            (ImageTaskGenerator(), 0.9),
        ]

        self.make_work_dir()

    def make_work_dir(self):
        if not os.path.exists(WORK_DIR):
            os.makedirs(WORK_DIR)
            bt.logging.info(f"Created work directory at {WORK_DIR}")

    async def query_miners(self):
        try:
            if len(self.synthetic_history) > MAX_SYNTHETIC_HISTORY_SIZE:
                return

            if not self.synthetic_tasks:
                return

            task, synapse = self.synthetic_tasks.pop(0)
            bt.logging.info("Popping synthetic task and sending it to miners")

            miner_uids = get_random_uids(self.neuron, k=self.config.neuron.sample_size)        
            bt.logging.debug(f"Selected miner uids: {miner_uids}")

            async with bt.dendrite(wallet=self.neuron.wallet) as dendrite:
                all_synapse_results = await dendrite(
                    axons = [self.neuron.metagraph.axons[uid] for uid in miner_uids],
                    synapse=synapse,
                    timeout=task.timeout
                )

            solutions = []

            for synapse, miner_uid in zip(all_synapse_results, miner_uids):
                processed_synapse = await self.process_synapse(synapse)
                if processed_synapse is not None:
                    solutions.append(Solution(html = processed_synapse.html, miner_uid = miner_uid, process_time = processed_synapse.dendrite.process_time))
                else:
                    self.un_responsed_count[miner_uid] += 1
                    
            if not solutions:
                bt.logging.warning(f"No valid solutions received")
                return
            bt.logging.info(f"Received {len(solutions)} solutions")

            self.synthetic_history.append((task, solutions))
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            raise e

    def update_raw_scores(self, rewards: List[float], miner_uids: List[int]):
        for i in range(len(miner_uids)):
            self.neuron.raw_scores[miner_uids[i]] += rewards[i]
        self.neuron.step += 1

        if self.neuron.step % UPDATE_SCORE_STEPS == 0:
            rewards_array = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
            rewards_array[:] = self.neuron.raw_scores[:] ** 3
            
            bt.logging.success(f"Blockchain rewards: {rewards_array}")
            self.neuron.update_scores(rewards_array, range(self.neuron.metagraph.n))
            self.neuron.raw_scores[:] = 0

    async def score(self):
        if len(self.synthetic_history) < MIN_SYNTHETIC_HISTORY_SIZE_TO_SCORE:
            return 
    
        task, solutions = random.choice(self.synthetic_history)
        task_generator = task.generator
        miner_uids = [solution.miner_uid for solution in solutions]
        
        rewards = await task_generator.reward(task, solutions)
        bt.logging.success(f"Rewards for {miner_uids}: {rewards}")
        self.update_raw_scores(rewards, miner_uids)
        self.synthetic_history = []

    async def synthensize_task(self):
        try:
            if len(self.synthetic_tasks) > MAX_SYNTHETIC_TASK_SIZE:
                return

            bt.logging.debug(f"Synthensize task")
            
            task_generator, _ = random.choices(
                self.task_generators,
                weights=[weight for _, weight in self.task_generators]
            )[0]
            
            task, synapse = await task_generator.generate_task()
            self.synthetic_tasks.append((task, synapse))
        except Exception as e:
            bt.logging.error(f"Error in synthensize_task: {e}")

    async def organic_forward(self, synapse: Union[WebgenieTextSynapse, WebgenieImageSynapse]):
        if isinstance(synapse, WebgenieTextSynapse):
            bt.logging.debug(f"Organic text forward: {synapse.prompt}")
        else:
            bt.logging.debug(f"Organic image forward: {synapse.base64_image[:MAX_DEBUG_IMAGE_STRING_LENGTH]}...")

        best_miner_uid = 3
        try:
            axon = self.neuron.metagraph.axons[best_miner_uid]
            async with bt.dendrite(wallet=self.neuron.wallet) as dendrite:
                responses = await dendrite(
                    axons=[axon],
                    synapse=synapse,
                    timeout=synapse.timeout,
                )

            processed_synapse = await self.process_synapse(responses[0])
            if processed_synapse is None:
                raise Exception(f"No valid solution received")
 
            return processed_synapse
        except Exception as e:
            bt.logging.error(f"[forward_organic_synapse] Error querying dendrite: {e}")
            synapse.html = f"Error: {e}"
            return synapse
    
    async def process_synapse(self, synapse: bt.Synapse) -> bt.Synapse:
        if synapse.dendrite.status_code == 200:
            synapse.html = preprocess_html(synapse.html)
            if not synapse.html:
                return None
            return synapse
        return None
