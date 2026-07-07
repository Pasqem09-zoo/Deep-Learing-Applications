"""
Main entry point for running reinforcement learning experiments.

This script:
- loads the configuration from config.py
- creates separate training and evaluation environments
- initializes the policy and value networks
- runs the selected algorithm: REINFORCE or A2C
- optionally renders the best saved policy at the end
"""

import os
import gymnasium as gym
import torch
import torch.optim as optim
import config

from src.utils import set_seed
from src.networks import PolicyNet, ValueNet
from src.reinforce import train_reinforce, run_episode
from src.a2c import A2CTrainer
from src.logger import get_logger




def main():

    set_seed(config.SEED)

    logger = get_logger(config)

    # Create separate Gymnasium environments for training and evaluation
    env_train = gym.make(config.ENV_NAME, max_episode_steps=config.MAX_STEPS)
    env_eval = gym.make(
        config.ENV_NAME,
        max_episode_steps=config.MAX_STEPS,
    )

    state_dim = env_train.observation_space.shape[0]
    action_dim = env_train.action_space.n

    policy = PolicyNet(state_dim, action_dim, config.HIDDEN_DIM)

    env_id = config.ENV_NAME.lower().replace("-", "_")
    checkpoint_dir = "lab3/checkpoints"
    checkpoint_prefix = f"{env_id}_{config.ALGORITHM}"
    best_checkpoint = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_best.pt")

    if config.ALGORITHM == "a2c":
        critic = ValueNet(state_dim, config.HIDDEN_DIM)
        actor_optim = optim.Adam(policy.parameters(), lr=config.A2C_ACTOR_LR)
        critic_optim = optim.Adam(critic.parameters(), lr=config.A2C_CRITIC_LR)

        trainer = A2CTrainer(
            env=env_train,
            actor=policy,
            critic=critic,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            config=config,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=checkpoint_prefix,
            log_callback=logger.log,
            eval_env=env_eval,
        )
        episode_rewards = trainer.train()
    else:
        value_net = None
        value_optimizer = None
        if config.USE_VALUE_BASELINE:
            value_net = ValueNet(state_dim, config.HIDDEN_DIM)
            value_optimizer = optim.Adam(
                value_net.parameters(),
                lr=config.VALUE_LEARNING_RATE
            )

        optimizer = optim.Adam(
            policy.parameters(),
            lr=config.LEARNING_RATE
        )

        episode_rewards, eval_rewards, eval_lengths = train_reinforce(
            env=env_train,
            eval_env=env_eval,
            policy=policy,
            optimizer=optimizer,
            episodes=config.EPISODES,
            max_steps=config.MAX_STEPS,
            gamma=config.GAMMA,
            checkpoint_every=config.CHECKPOINT_EVERY,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=checkpoint_prefix,
            N=config.N,
            M=config.M,
            log_callback=logger.log,
            value_net=value_net,
            value_optimizer=value_optimizer,
        )



    if config.RENDER:
        if os.path.exists(best_checkpoint):
            policy.load_state_dict(torch.load(best_checkpoint))
        else:
            print(f"Checkpoint not found: {best_checkpoint}")
        render_env = gym.make(
            config.ENV_NAME,
            render_mode="human",
            max_episode_steps=config.MAX_STEPS,
        )

        if config.ENV_NAME.lower().startswith("cartpole"):
            render_env.unwrapped.theta_threshold_radians = config.EVAL_CARTPOLE_THETA
        policy.eval()

        for _ in range(config.RENDER_EPISODES):
            run_episode(render_env, policy, max_steps=config.MAX_STEPS)
        render_env.close()

    env_train.close()
    env_eval.close()
    logger.finish()


if __name__ == "__main__":
    main()