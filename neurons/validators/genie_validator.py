import bittensor as bt
import random
from typing import Union

from webgenie.base.neuron import BaseNeuron
from webgenie.constants import MAX_SYNTHETIC_HISTORY_SIZE, MAX_SYNTHETIC_TASK_SIZE
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

        self.task_generators = [
        #    TextTaskGenerator(),
            ImageTaskGenerator(),
        ]

        self.make_work_dir()

    def make_work_dir(self):
        import os
        from webgenie.constants import WORK_DIR
        
        if not os.path.exists(WORK_DIR):
            os.makedirs(WORK_DIR)
            bt.logging.info(f"Created work directory at {WORK_DIR}")

    async def forward(self):
        bt.logging.debug(f"Forward")
        try:
            if len(self.synthetic_history) > MAX_SYNTHETIC_HISTORY_SIZE:
                return

            if not self.synthetic_tasks:
                bt.logging.warning(f"No synthetic tasks")
                return
            bt.logging.debug(f"Synthetic tasks: {self.synthetic_tasks}")
            task, synapse = self.synthetic_tasks.pop(0)
            miner_uids = get_random_uids(self.neuron, k=self.config.neuron.sample_size)        
            bt.logging.debug(f"Selected miner uids: {miner_uids}")

            all_synapse_results = await self.neuron.dendrite(
                axons = [self.neuron.metagraph.axons[uid] for uid in miner_uids],
                synapse=synapse,
                timeout=task.timeout
            )

            solutions = []
            for synapse, miner_uid in zip(all_synapse_results, miner_uids):
                processed_synapse = await self.process_synapse(synapse)
                if processed_synapse is not None:
                    solutions.append(Solution(html = processed_synapse.html, miner_uid = miner_uid, process_time = processed_synapse.dendrite.process_time))

            if not solutions:
                bt.logging.warning(f"No valid solutions received")
                return

            bt.logging.debug(f"Processed solutions: {solutions}")
            self.synthetic_history.append((task, solutions))
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            raise e

    async def score(self):
        if not self.synthetic_history:
            bt.logging.warning(f"No synthetic history to score")
            return 
        
        task, solutions = self.synthetic_history.pop(0)
        task_generator = task.generator
        scores = await task_generator.reward(task, solutions)
        miner_uids = [solution.miner_uid for solution in solutions]
        bt.logging.debug(f"Miner uids: {miner_uids}")
        bt.logging.debug(f"Scores: {scores}")
        
        self.neuron.update_scores(scores, miner_uids)
        self.neuron.sync()

    async def synthensize_task(self):
        bt.logging.debug(f"Synthensize task")
        try:
            if len(self.synthetic_tasks) > MAX_SYNTHETIC_TASK_SIZE:
                return

            task, synapse = await random.choice(self.task_generators).generate_task()
            self.synthetic_tasks.append((task, synapse))
        except Exception as e:
            bt.logging.error(f"Error in synthensize_task: {e}")

    async def organic_forward(self, synapse: Union[WebgenieTextSynapse, WebgenieImageSynapse]):
        bt.logging.debug(f"Organic forward: {synapse}")
        best_miner_uid = 1
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
