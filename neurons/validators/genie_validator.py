import os
import bittensor as bt
import asyncio
import numpy as np
import random
from typing import Union, List

from webgenie.base.neuron import BaseNeuron
from webgenie.constants import (
    NUM_CONCURRENT_QUERIES,
    MAX_COMPETETION_HISTORY_SIZE, 
    MAX_SYNTHETIC_TASK_SIZE, 
    MAX_DEBUG_IMAGE_STRING_LENGTH, 
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
        self.competetions = []
        self.synthetic_tasks = []

        self.task_generators = [
            (TextTaskGenerator(), 0.5),
            (ImageTaskGenerator(), 0.5),
        ]

        self.make_work_dir()

    def make_work_dir(self):
        if not os.path.exists(WORK_DIR):
            os.makedirs(WORK_DIR)
            bt.logging.info(f"Created work directory at {WORK_DIR}")

    async def query_one_task(self, task, synapse, miner_uids):
        try:    
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
            
            return task, solutions
        except Exception as e:
            bt.logging.error(f"Error in query_one_task: {e}")
            raise e

    async def query_miners(self):
        try:
            if len(self.competetions) > MAX_COMPETETION_HISTORY_SIZE:
                return

            if len(self.synthetic_tasks) < NUM_CONCURRENT_QUERIES:
                return
                
            bt.logging.info("querying miners")

            miner_uids = get_random_uids(self.neuron, k=self.config.neuron.sample_size)
            bt.logging.debug(f"Selected miner uids: {miner_uids}")

            query_coroutines = [self.query_one_task(task, synapse, miner_uids) for task, synapse in self.synthetic_tasks]

            self.synthetic_tasks = []
            
            results = await asyncio.gather(*query_coroutines, return_exceptions=True)
            self.competetions.append(results)
        
        except Exception as e:
            bt.logging.error(f"Error in query_miners: {e}")
            raise e

    async def score(self):
        if not self.competetions:
            return

        results = self.competetions.pop(0)
        tatal_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        for result in results:
            if isinstance(result, Exception):
                continue

            task, solutions = result
            if not solutions:
                continue

            task_generator = task.generator
            miner_uids = [solution.miner_uid for solution in solutions]
            rewards = await task_generator.reward(task, solutions)
            bt.logging.success(f"Rewards for {miner_uids}: {rewards}")
            
            for i in range(len(miner_uids)):
                tatal_scores[miner_uids[i]] += rewards[i]

        tatal_scores[:] = tatal_scores[:] ** 3
        self.neuron.update_scores(tatal_scores, range(self.neuron.metagraph.n))
        self.neuron.step += 1

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
