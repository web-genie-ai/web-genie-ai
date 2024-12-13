import bittensor as bt
import os
import wandb

import webgenie

wandb_on = False

def init_wandb(self):
    try:
        global wandb_on
        if self.config.wandb.off:
            wandb_on = False
            return

        wandb_on = True
        wandb.login(key=os.getenv("WANDB_API_KEY"))

        run_name = f"{self.config.neuron.name}-{self.uid}"
        run = wandb.init(
            project=webgenie.PROJECT_NAME, 
            entity=os.getenv("WANDB_ENTITY_NAME"), 
            name=run_name,
            config=self.config,
            reinit=True
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
        if not wandb_on:
            return
        wandb.log(data)
    except Exception as e:
        bt.logging.error(f"Error logging to wandb: {e}")
        raise e

