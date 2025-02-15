import bittensor as bt
import copy
import numpy as np

from io import StringIO
from rich.console import Console
from rich.table import Table
from typing import List

from webgenie.base.neuron import BaseNeuron
from webgenie.challenges.challenge import Challenge
from webgenie.constants import CONSIDERING_SESSION_COUNTS, __STATE_VERSION__, WORK_DIR
from webgenie.helpers.weights import save_file_to_wandb

class ScoreManager:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron
        self.state_path = self.neuron.config.neuron.full_path + "/state.npz"
        self.lock = neuron.lock

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

        # Check to see if the metagraph has changed size.
        # If so, we need to add new hotkeys and moving averages.
        if len(self.hotkeys) < len(new_hotkeys):
            new_total_scores = np.zeros((len(new_hotkeys)))
            min_len = min(len(self.hotkeys), len(self.total_scores))
            new_total_scores[:min_len] = self.total_scores[:min_len]
            self.total_scores = new_total_scores

        # Update the hotkeys.
        self.hotkeys = copy.deepcopy(new_hotkeys)
        with self.lock:
            self.save_scores()

    def update_scores(self, rewards: np.ndarray, uids: List[int], challenge: Challenge):
        bt.logging.info("Updating scores")
        session = challenge.session
        competition_type = challenge.competition_type
        if self.current_session != session:
            # This is a new session, reset the scores and winners.
            self.current_session = session
            self.number_of_tasks = 0
            self.total_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        # Update accumulated scores and track best performer
        self.number_of_tasks += 1
        self.total_scores[uids] += rewards

        current_session_results = {
            "session": session,
            "competition_type": competition_type,
            "number_of_tasks": self.number_of_tasks,
            "winner": np.argmax(self.total_scores),
            "scores": self.total_scores,
        }

        self.session_results[session] = current_session_results
        for session_number in list(self.session_results.keys()):
            if session_number < session - CONSIDERING_SESSION_COUNTS * 2:
                self.session_results.pop(session_number)
 
        with self.lock:
            self.save_scores()

        console = Console()
        self.print_session_result(session, console)

    def get_scores(self, session_upto: int):
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
        session_result = self.session_results[session_upto]

        number_of_tasks = session_result["number_of_tasks"]
        session = session_result["session"]
        competition_type = session_result["competition_type"]
        winner = session_result["winner"]
        scores = session_result["scores"]

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
        total_scores_table.add_column("Total Score", justify="right", style="green")
        total_scores_table.add_column("Average Score", justify="right", style="yellow")
        scored_uids = [(uid, score) for uid, score in enumerate(scores) if score > 0]
        scored_uids.sort(key=lambda x: x[1], reverse=True)
        for rank, (uid, score) in enumerate(scored_uids):
            total_scores_table.add_row(
                str(rank + 1),
                str(uid),
                f"{score:.4f}",
                f"{score / number_of_tasks:.4f}",
            )
        console.print(total_scores_table)

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
