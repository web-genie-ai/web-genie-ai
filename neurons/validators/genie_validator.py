import os
import bittensor as bt
import random
from typing import Union

from webgenie.base.neuron import BaseNeuron
from webgenie.constants import (
    MAX_COMPETETION_HISTORY_SIZE, 
    MAX_SYNTHETIC_TASK_SIZE, 
    WORK_DIR,
)
from webgenie.competitions import (
    ImageTaskAccuracyCompetition, 
    ImageTaskQualityCompetition,
    ImageTaskSeoCompetition,
    RESERVED_WEIGHTS,
    ACCURACY_METRIC_NAME,
    QUALITY_METRIC_NAME,
    SEO_METRIC_NAME,
)
from webgenie.storage import (
    upload_competition,
    upload_competition_result,
)
from webgenie.helpers.htmls import preprocess_html, validate_resources
from webgenie.helpers.images import image_debug_str
from webgenie.protocol import WebgenieImageSynapse, WebgenieTextSynapse
from webgenie.tasks import Solution
from webgenie.utils.uids import get_all_available_uids, get_most_available_uid


class GenieValidator:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron
        self.config = neuron.config
        self.competitions = []
        self.synthetic_tasks = []

        self.avail_competitions = [
            (ImageTaskAccuracyCompetition(), 0.4),
            (ImageTaskSeoCompetition(), 0.3),
            (ImageTaskQualityCompetition(), 0.3),
        ]

        self.make_work_dir()

    def make_work_dir(self):
        if not os.path.exists(WORK_DIR):
            os.makedirs(WORK_DIR)
            bt.logging.info(f"Created work directory at {WORK_DIR}")

    async def query_miners(self):
        try:
            if len(self.competitions) > MAX_COMPETETION_HISTORY_SIZE:
                return

            if not self.synthetic_tasks:
                return
                
            bt.logging.info("querying miners")

            task, synapse = self.synthetic_tasks.pop(0)
            miner_uids = get_all_available_uids(self.neuron)
            
            async with bt.dendrite(wallet=self.neuron.wallet) as dendrite:
                all_synapse_results = await dendrite(
                    axons = [self.neuron.metagraph.axons[uid] for uid in miner_uids],
                    synapse=synapse,
                    timeout=task.timeout,
                )

            solutions = []
            for synapse, miner_uid in zip(all_synapse_results, miner_uids):
                processed_synapse = await self.process_synapse(synapse)
                if processed_synapse is not None:
                    solutions.append(
                        Solution(
                            html = processed_synapse.html, 
                            miner_uid = miner_uid, 
                            process_time = processed_synapse.dendrite.process_time,
                        )
                    )
            
            self.competitions.append((task, solutions))
        except Exception as e:
            bt.logging.error(f"Error in query_miners: {e}")
            raise e

    async def score(self):
        if not self.competitions:
            return

        task, solutions = self.competitions.pop(0)
        if not solutions:
            return

        best_miner = -1
        best_final_score = 0.0
        
        solutions.sort(key=lambda solution: solution.process_time)
        competition = task.competition
        upload_competition({
            "competition_type": competition.COMPETITION_TYPE,
            "task": task,
        })

        miner_uids = [solution.miner_uid for solution in solutions]
        
        final_scores, scores = await competition.calculate_final_scores(task, solutions)
        bt.logging.success(f"Final scores for {miner_uids}: {final_scores}")
        
        for i in range(len(miner_uids)):
            upload_competition_result({
                "miner_uid": miner_uids[i],
                "final_score": final_scores[i],
                "accuracy": scores[ACCURACY_METRIC_NAME][i],
                "quality": scores[QUALITY_METRIC_NAME][i],
                "seo": scores[SEO_METRIC_NAME][i],
                "html": solutions[i].html,
            })
            
            if final_scores[i] > best_final_score:
                best_final_score = final_scores[i]
                best_miner = miner_uids[i]

        if best_miner == -1:
            return
    
        self.neuron.update_scores([RESERVED_WEIGHTS[competition.COMPETITION_TYPE]], [best_miner])
        self.neuron.step += 1

    async def synthensize_task(self):
        try:
            if len(self.synthetic_tasks) > MAX_SYNTHETIC_TASK_SIZE:
                return

            bt.logging.debug(f"Synthensize task")
            
            competition, _ = random.choices(
                self.avail_competitions,
                weights=[weight for _, weight in self.avail_competitions],
            )[0]
            
            task, synapse = await competition.generate_task()
            self.synthetic_tasks.append((task, synapse))
        except Exception as e:
            bt.logging.error(f"Error in synthensize_task: {e}")

    async def organic_forward(self, synapse: Union[WebgenieTextSynapse, WebgenieImageSynapse]):
        if isinstance(synapse, WebgenieTextSynapse):
            bt.logging.debug(f"Organic text forward: {synapse.prompt}")
        else:
            bt.logging.debug(f"Organic image forward: {image_debug_str(synapse.base64_image)}...")

        best_miner_uid = get_most_available_uid(self.neuron)
        all_miner_uids = get_all_available_uids(self.neuron)
        try:
            async with bt.dendrite(wallet=self.neuron.wallet) as dendrite:
                responses = await dendrite(
                    axons=[self.neuron.metagraph.axons[uid] for uid in all_miner_uids],
                    synapse=synapse,
                    timeout=synapse.timeout,
                )

            processed_synapse = await self.process_synapse(responses[best_miner_uid])
            if processed_synapse is None:
                raise Exception(f"No valid solution received")
 
            return processed_synapse
        except Exception as e:
            bt.logging.error(f"[forward_organic_synapse] Error querying dendrite: {e}")
            synapse.html = f"Error: {e}"
            return synapse
    
    async def process_synapse(self, synapse: bt.Synapse) -> bt.Synapse:
        if synapse.dendrite.status_code == 200:
            html = preprocess_html(synapse.html)
            if not html:
                return None
            if not validate_resources(html):
                return None
            synapse.html = html
            return synapse
        return None
