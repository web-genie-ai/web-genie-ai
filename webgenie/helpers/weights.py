import bittensor as bt
import wandb

from webgenie.constants import (
    WANDB_OFF,
    WANDB_API_KEY,
    WANDB_PROJECT_NAME,
    WANDB_ENTITY_NAME,
)


def init_wandb(self):
    try:
        if WANDB_OFF:
            return
        wandb.login(key=WANDB_API_KEY)
        run_name = f"{self.config.neuron.name}-{self.uid}"
        run = wandb.init(
            project=WANDB_PROJECT_NAME, 
            entity=WANDB_ENTITY_NAME, 
            name=run_name,
            config=self.config,
            reinit=True,
        )
        signature = self.wallet.hotkey.sign(run.id.encode()).hex()
        self.config.signature = signature
        wandb.config.update(self.config , allow_val_change=True)
        bt.logging.success(f"Wandb initialized with run id: {run.id}")
    except Exception as e:
        bt.logging.error(f"Error initializing wandb: {e}")
        raise e


def log_wandb(data: dict):
    try:
        if WANDB_OFF:
            return
        wandb.log(data)
    except Exception as e:
        bt.logging.error(f"Error logging to wandb: {e}")
        raise e