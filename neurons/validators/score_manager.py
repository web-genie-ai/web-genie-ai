import bittensor as bt
import copy
import numpy as np
import threading
from io import StringIO
from rich.console import Console
from rich.table import Table
from typing import List

from webgenie.base.neuron import BaseNeuron

from webgenie.challenges.challenge import Challenge, RESERVED_WEIGHTS
from webgenie.constants import (
    CONSIDERING_SESSION_COUNTS,
    __STATE_VERSION__,
    WORK_DIR,
    MAX_UNANSWERED_TASKS
)
from webgenie.helpers.weights import save_file_to_wandb
from webgenie.storage import submit_results
class ScoreManager:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron
        self.state_path = self.neuron.config.neuron.full_path + "/state.npz"
        self.lock = self.neuron.lock

        self.hotkeys = copy.deepcopy(self.neuron.metagraph.hotkeys)
        self.current_session = -1
        self.number_of_tasks = 0
        self.total_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        self.last_set_weights_session = -1
        self.session_results = {}

    def load_scores(self):
        try:
            bt.logging.info(f"Loading scores from {self.state_path}")
            data = np.load(self.state_path, allow_pickle=True)

            self.hotkeys = data.get(
                f"hotkeys", 
                copy.deepcopy(self.neuron.metagraph.hotkeys)
            )
            
            self.current_session = data.get(
                f"current_session", 
                -1
            )

            self.number_of_tasks = data.get(
                f"number_of_tasks", 
                0
            )
            
            self.last_set_weights_session = data.get(
                f"last_set_weights_session", 
                -1
            )

            self.solved_tasks = data.get(
                f"solved_tasks", 
                np.zeros(self.neuron.metagraph.n, dtype=np.float32),
            )

            self.total_scores = data.get(
                f"total_scores_{__STATE_VERSION__}", 
                np.zeros(self.neuron.metagraph.n, dtype=np.float32),
            )
            
            self.session_results = dict(
                data.get("session_results", np.array({})).item()
            )
        except Exception as e:
            bt.logging.error(f"Error loading state: {e}")
            self.hotkeys = copy.deepcopy(self.neuron.metagraph.hotkeys)
            self.current_session = -1
            self.solved_tasks = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
            self.total_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
            self.last_set_weights_session = -1
            self.number_of_tasks = 0
            self.session_results = {}

    def save_scores(self):
        try:
            bt.logging.info(f"Saving scores to {self.state_path}")
            np.savez(
                self.state_path,
                hotkeys=self.hotkeys,
                **{f"current_session": self.current_session},
                last_set_weights_session=self.last_set_weights_session,
                number_of_tasks=self.number_of_tasks,
                solved_tasks=self.solved_tasks,
                **{f"total_scores_{__STATE_VERSION__}": self.total_scores},
                session_results= self.session_results,
                allow_pickle=True,
            )
        except Exception as e:
            bt.logging.error(f"Error saving state: {e}")
    
    def set_new_hotkeys(self, new_hotkeys: List[str]):
        bt.logging.info(
            "Hotkeys updated, re-syncing scores"
        )
        # Zero out all hotkeys that have been replaced.
        for uid, hotkey in enumerate(self.hotkeys):
            if hotkey != new_hotkeys[uid]:
                self.total_scores[uid] = 0
                self.solved_tasks[uid] = 0

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(new_hotkeys):
            new_total_scores = np.zeros((len(new_hotkeys)))
            min_len = min(len(self.hotkeys), len(self.total_scores))
            new_total_scores[:min_len] = self.total_scores[:min_len]
            self.total_scores = new_total_scores

            new_solved_tasks = np.zeros((len(new_hotkeys)))
            min_len = min(len(self.hotkeys), len(self.solved_tasks))
            new_solved_tasks[:min_len] = self.solved_tasks[:min_len]
            self.solved_tasks = new_solved_tasks

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(new_hotkeys)
        self.save_scores()

    def update_scores(self, rewards: np.ndarray, uids: List[int], challenge: Challenge):
        bt.logging.info("Updating scores")
        session = challenge.session
        competition_type = challenge.competition_type
        if self.current_session != session:
            # This is a new session, reset the scores and winners.
            self.current_session = session
            self.number_of_tasks = 0
            self.solved_tasks = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
            self.total_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        # Update accumulated scores and track best performer
        self.number_of_tasks += 1
        self.total_scores[uids] += rewards
        self.solved_tasks[uids] += 1

        winner = self.get_winner(
            self.total_scores,
            self.solved_tasks,
            self.number_of_tasks,
        )
        
        current_session_results = {
            "session": session,
            "competition_type": competition_type,
            "number_of_tasks": self.number_of_tasks,
            "winner": winner,
            "solved_tasks": self.solved_tasks,
            "scores": self.total_scores,
        }

        self.session_results[session] = current_session_results
        for session_number in list(self.session_results.keys()):
            if session_number < session - CONSIDERING_SESSION_COUNTS * 2:
                self.session_results.pop(session_number)
 
        self.save_scores()

        console = Console()
        self.print_session_result(session, console)
    
    def is_blacklisted(self, uid: int):
        blacklisted_coldkeys = ["5G9yTkkDd39chZiyvKwNsQvzqbbPgdiLtdb4sCR743f4MuRY"]
        return self.neuron.metagraph.axons[uid].coldkey in blacklisted_coldkeys

    def get_winner(self, total_scores: np.ndarray, solved_tasks: np.ndarray, number_of_tasks: int):
        avg_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        for uid in range(self.neuron.metagraph.n):
            if self.is_blacklisted(uid):
                continue
            
            avg_scores[uid] = total_scores[uid] / number_of_tasks
            
            # if solved_tasks[uid] >= max(1, number_of_tasks - MAX_UNANSWERED_TASKS):
            #     avg_scores[uid] = total_scores[uid] / solved_tasks[uid]
            # else:
            #     avg_scores[uid] = 0
        winner = np.argmax(avg_scores) if max(avg_scores) > 0 else -1
        return winner

    def get_scores(self, session_upto: int):
        # scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        # for session_number in self.session_results:
        #     if (session_number <= session_upto - CONSIDERING_SESSION_COUNTS or 
        #         session_number > session_upto):
        #         continue

        #     try:
        #         winner = self.session_results[session_number]["winner"]
        #         competition_type = self.session_results[session_number]["competition_type"]
        #         if winner == -1:
        #             continue
        #         scores[winner] += RESERVED_WEIGHTS[competition_type]
        #     except Exception as e:
        #         bt.logging.warning(f"Error getting scores: {e}")

        # return scores
        scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        tiny_weight = 1 / 128
        big_weight = 1.0
        for session_number in self.session_results:
            if (session_number <= session_upto - CONSIDERING_SESSION_COUNTS or 
                session_number > session_upto):
                continue
                
            winner = self.session_results[session_number]["winner"]
            if winner == -1:
                continue
            if session_number == session_upto:
                scores[winner] += big_weight
            else:
                scores[winner] += tiny_weight
        return scores
        
        # if session_upto in self.session_results:
        #     scores = self.session_results[session_upto]["scores"]
        # else:
        #     scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        # return np.power(scores, 9)

    def print_session_result(self, session_upto: int, console: Console):
        try:
            session_result = self.session_results[session_upto]

            number_of_tasks = session_result["number_of_tasks"]
            session = session_result["session"]
            competition_type = session_result["competition_type"]
            winner = session_result["winner"]
            scores = session_result["scores"]
            solved_tasks = session_result["solved_tasks"]
            
            avg_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
            for uid in range(self.neuron.metagraph.n):
                if solved_tasks[uid] >= max(1, number_of_tasks - MAX_UNANSWERED_TASKS):
                    avg_scores[uid] = scores[uid] / solved_tasks[uid]
                else:
                    avg_scores[uid] = 0
            
            total_scores_table = Table(
                title=(
                    f"📊 Total Scores Summary\n"
                    f"🔄 Session: #{session}\n"
                    f"📝 Number of Tasks: #{number_of_tasks}\n" 
                    f"🏆 Competition: {competition_type}\n"
                    f"👑 Winner: #{winner}\n"
                ),
                show_header=True,
                header_style="bold magenta", 
                title_style="bold blue",
                border_style="blue"
            )

            total_scores_table.add_column("Rank", justify="right", style="red", header_style="bold red")
            total_scores_table.add_column("UID", justify="right", style="cyan", header_style="bold cyan")
            total_scores_table.add_column("Average Score", justify="right", style="yellow")
            scored_uids = [(uid, avg_scores[uid]) for uid in range(self.neuron.metagraph.n) if avg_scores[uid] > 0]
            scored_uids.sort(key=lambda x: x[1], reverse=True)
            for rank, (uid, score) in enumerate(scored_uids):
                total_scores_table.add_row(
                    str(rank + 1),
                    str(uid),
                    f"{score:.4f}",
                )
            console.print(total_scores_table)
        except Exception as e:
            bt.logging.warning(f"Error printing session result: {e}")

    def save_session_result_to_file(self, session_upto: int):
        try:
            log_file_name = f"{WORK_DIR}/session_{session_upto}.txt"
            console = Console(file=StringIO(), force_terminal=False)
            self.print_session_result(session_upto, console)
            table_str = console.file.getvalue()
            with open(log_file_name, "w") as f:
                f.write(table_str)
            save_file_to_wandb(log_file_name)
        except Exception as e:
            bt.logging.error(f"Error saving session result to file: {e}")
            raise e

    def submit_results_to_dashboard(self, session_upto: int):
        try:
            session_result = self.session_results[session_upto]

            number_of_tasks = session_result["number_of_tasks"]
            session = session_result["session"]
            competition_type = session_result["competition_type"]
            scores = session_result["scores"]
            solved_tasks = session_result["solved_tasks"]            
            competition = {
                "session_number": int(session),
                "competition_type": competition_type,
            }
            
            submissions = []   
            for uid in range(self.neuron.metagraph.n):
                if solved_tasks[uid] < max(1, number_of_tasks - MAX_UNANSWERED_TASKS):
                    continue
                avg_score = scores[uid] / solved_tasks[uid]
                submissions.append({
                    "neuron": {
                        "hotkey": self.neuron.metagraph.hotkeys[uid],
                    },
                    "score": float(avg_score),
                })

            submit_results({
                "competition": competition,
                "submissions": submissions,
            })
        except Exception as e:
            bt.logging.error(f"Error submitting results to dashboard: {e}")
            raise e
