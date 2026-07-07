"""
Simple W&B logger wrapper to keep main.py clean.
"""

import wandb


class Logger:
	def __init__(self, enabled):
		self.enabled = enabled

	def log(self, data):
		if self.enabled:
			wandb.log(data)

	def finish(self):
		if self.enabled:
			wandb.finish()


def get_logger(config):
	if not config.WANDB_ENABLED:
		return Logger(False)

	wandb_kwargs = {"project": config.WANDB_PROJECT}
	if config.WANDB_RUN_NAME:
		wandb_kwargs["name"] = config.WANDB_RUN_NAME
	wandb.init(
    **wandb_kwargs,
    config={
        "env_name": config.ENV_NAME,
        "algorithm": config.ALGORITHM,
        "episodes": config.EPISODES,
        "max_steps": config.MAX_STEPS,
        "gamma": config.GAMMA,
        "hidden_dim": config.HIDDEN_DIM,
        "a2c_n_steps": config.A2C_N_STEPS,
        "a2c_entropy_beta": config.A2C_ENTROPY_BETA,
        "a2c_adv_norm": config.A2C_ADV_NORM,
    	}
	)
	return Logger(True)
