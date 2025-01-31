import bittensor as bt
import copy
import numpy as np

from rich.console import Console
from rich.table import Table
from typing import List

from webgenie.base.neuron import BaseNeuron
from webgenie.challenges.challenge import Challenge, RESERVED_WEIGHTS
from webgenie.constants import CONSIDERING_SESSION_COUNTS, __STATE_VERSION__


class ScoreManager:
    def __init__(self, neuron: BaseNeuron):
        self.neuron = neuron
        self.state_path = self.neuron.config.neuron.full_path + "/state.npz"
        self.lock = neuron.lock

        self.hotkeys = copy.deepcopy(self.neuron.metagraph.hotkeys)
        self.current_session = -1
        self.total_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        self.last_set_weights_session = -1
        self.winners = {}

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
            
            self.last_set_weights_session = data.get(
                f"last_set_weights_session", 
                -1
            )

            self.total_scores = data.get(
                f"total_scores_{__STATE_VERSION__}", 
                np.zeros(self.neuron.metagraph.n, dtype=np.float32),
            )
            
            self.winners = dict(data.get(f"winners_{__STATE_VERSION__}", {}).item())
        except Exception as e:
            bt.logging.error(f"Error loading state: {e}")
            self.hotkeys = copy.deepcopy(self.neuron.metagraph.hotkeys)
            self.current_session = -1
            self.total_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
            self.last_set_weights_session = -1
            self.winners = {}

    def save_scores(self):
        try:
            bt.logging.info(f"Saving scores to {self.state_path}")
            np.savez(
                self.state_path,
                hotkeys=self.hotkeys,
                **{f"current_session": self.current_session},
                last_set_weights_session=self.last_set_weights_session,
                **{f"total_scores_{__STATE_VERSION__}": self.total_scores},
                **{f"winners_{__STATE_VERSION__}": self.winners},
                allow_pickle=True,
            )
            self.should_save = False
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
            self.total_scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        # Update accumulated scores and track best performer
        self.total_scores[uids] += rewards
        # Create a rich table to display total scores
  
        total_scores_table = Table(
            title=f"Total Scores This Session-{session}",
            show_header=True,
            header_style="bold magenta", 
            title_style="bold blue",
            border_style="blue"
        )
        total_scores_table.add_column("UID", justify="right", style="cyan", header_style="bold cyan")
        total_scores_table.add_column("Total Score", justify="right", style="green")
        
        # Add rows for non-zero scores, sorted by score
        scored_uids = [(uid, score) for uid, score in enumerate(self.total_scores) if score > 0]
        scored_uids.sort(key=lambda x: x[1], reverse=True)
        
        for uid, score in scored_uids:
            total_scores_table.add_row(
                str(uid),
                f"{score:.4f}"
            )

        console = Console()
        console.print(total_scores_table)

        if np.max(self.total_scores) > 0:
            self.winners[session] = (np.argmax(self.total_scores), competition_type)
        else:
            self.winners[session] = (-1, competition_type)

        # Remove old winners
        for session_number in list(self.winners.keys()):
            if session_number < session - CONSIDERING_SESSION_COUNTS * 2:
                self.winners.pop(session_number)
 
        # Create a rich table to display the winners
        table = Table(
            title="Winners by Session",
            show_header=True,
            header_style="bold magenta",
            title_style="bold blue",
            border_style="blue"
        )
        table.add_column("Session", justify="right", style="cyan", header_style="bold cyan")
        table.add_column("Winner UID", justify="right", style="green")
        table.add_column("Competition Type", justify="left")
        # Add rows sorted by session number
        for session_number in sorted(self.winners.keys()):
            winner_uid, competition_type = self.winners[session_number]
            table.add_row(
                str(session_number),
                str(winner_uid),
                str(competition_type),
            )

        console = Console()
        console.print(table)
        with self.lock:
            self.save_scores()
        
    def get_scores(self, session_upto: int):
        #return self.total_scores
        scores = np.zeros(self.neuron.metagraph.n, dtype=np.float32)
        with self.lock:
            for session_number in self.winners:
                if (session_number <= session_upto - CONSIDERING_SESSION_COUNTS or 
                    session_number > session_upto):
                    continue
                
                winner, competition_type = self.winners[session_number]
                if winner == -1:
                    continue
                scores[winner] += RESERVED_WEIGHTS[competition_type]
        return scores
