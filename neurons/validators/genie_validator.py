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

            miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
        
            task, synapse = await random.choice(self.task_generators).generate_task()

            all_synapse_results = await self.neuron.dendrite(
                axons = [self.metagraph.axons[uid] for uid in miner_uids],
                synapse=synapse,
                timeout=task.timeout
            )

            processed_synapses, processed_miner_uids = await self.process_synapses(all_synapse_results, miner_uids)

            if len(processed_synapses) == 0:
                bt.logging.error(f"No valid synapses received")
                return
            bt.logging.debug(f"Processed synapses: {processed_synapses}")

            self.synthetic_history.append((task, processed_synapses, processed_miner_uids))
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            raise e

    async def process_synapses(self, synapses: List[WebgenieStreamingSynapse], miner_uids: List[int]):
        processed_synapses = []
        processed_miner_uids = []
        
        for synapse, miner_uid in zip(synapses, miner_uids):
            if synapse.dendrite.status_code == 200:
                processed_synapses.append(synapse)
                processed_miner_uids.append(miner_uid)

        return processed_synapses, processed_miner_uids

    async def score(self):
        if len(self.synthetic_history) == 0:
            bt.logging.warning(f"No synthetic history to score")
            return 
        
        task, synapses, miner_uids = self.synthetic_history.pop(0)

        task_generator = task.generator
        scores = await task_generator.reward(task, synapses)
        self.neuron.update_scores(scores, miner_uids)
        self.neuron.sync()

    async def organic_forward(self, synapse: WebgenieTextSynapse):
        axon = self.metagraph.axons[1]
        try:
            async with bt.dendrite(wallet=self.wallet) as dendrite:
                bt.logging.info(f"Dendrite: {dendrite}")
                responses = await dendrite(
                    axons=[axon],
                    synapse=synapse,
                    timeout=synapse.timeout,
                )
                return responses[0]
        except Exception as e:
            bt.logging.error(f"[forward_organic_synapse] Error querying dendrite: {e}")
            synapse.solution = Solution(
                html = "",
            )
            return synapse