import bittensor as bt
import random
import torch
from typing import List

from webgenie.base.neuron import BaseNeuron
from webgenie.protocol import WebgenieStreamingSynapse, WebgenieTextSynapse
from webgenie.rewards import RewardManager
from webgenie.solution import Solution
from webgenie.task_generator.image_task_generator import ImageTaskGenerator
from webgenie.task_generator.text_task_generator import TextTaskGenerator
from webgenie.utils.uids import get_random_uids


MAX_SYNTHETIC_HISTORY_SIZE = 10

class GenieValidator:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron
        self.config = neuron.config
        self.synthetic_history = []

        self.reward_manager = RewardManager()
        self.task_generators = [
            TextTaskGenerator(),
            ImageTaskGenerator(),
        ]

    async def forward(self):
        try:
            if len(self.synthetic_history) > MAX_SYNTHETIC_HISTORY_SIZE:
                return

            miner_uids = get_random_uids(self.neuron, k=self.config.neuron.sample_size)
        
            task, synapse = await random.choice(self.task_generators).generate_task()

            all_synapse_results = await self.neuron.dendrite(
                axons = [self.neuron.metagraph.axons[uid] for uid in miner_uids],
                synapse=synapse,
                timeout=task.timeout
            )

            solutions = []

            for synapse, miner_uid in zip(all_synapse_results, miner_uids):
                processed_synapse = await self.process_synapse(synapse, miner_uid)
                if processed_synapse is not None:
                    solutions.append(processed_synapse.solution)

            if len(solutions) == 0:
                bt.logging.warning(f"No valid solutions received")
                return
            
            bt.logging.debug(f"Processed solutions: {solutions}")

            self.synthetic_history.append((task, solutions))
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            raise e

    async def score(self):
        if len(self.synthetic_history) == 0:
            bt.logging.warning(f"No synthetic history to score")
            return 
        
        task, solutions = self.synthetic_history.pop(0)

        task_generator = task.generator
        scores = await task_generator.reward(task, solutions)
        self.neuron.update_scores(scores, [solution.miner_uid for solution in solutions])
        self.neuron.sync()

    async def organic_forward(self, synapse: WebgenieTextSynapse):
        best_miner_uid = 1
        try:
            axon = self.metagraph.axons[best_miner_uid]

            async with bt.dendrite(wallet=self.wallet) as dendrite:
                bt.logging.info(f"Dendrite: {dendrite}")
                responses = await dendrite(
                    axons=[axon],
                    synapse=synapse,
                    timeout=synapse.timeout,
                )
                
            processed_synapse = await self.process_synapse(responses[0], best_miner_uid)
            if processed_synapse is None:
                raise Exception(f"No valid solution received")
            
            return processed_synapse
        except Exception as e:
            bt.logging.error(f"[forward_organic_synapse] Error querying dendrite: {e}")
            synapse.solution = Solution(
                html = f"Error: {e}",
                process_time = 0,
                miner_uid = best_miner_uid,
            )
            return synapse
    
    async def process_synapse(self, synapse: bt.Synapse, miner_uid: int) -> bt.Synapse:
        if synapse.dendrite.status_code == 200:
            if synapse.solution is None:
                return None
            synapse.solution.miner_uid = miner_uid
            return synapse
        return None