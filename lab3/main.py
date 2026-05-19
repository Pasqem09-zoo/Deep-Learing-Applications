"""
Main entry point for running reinforcement learning experiments.

This file:
- loads the experiment configuration
- creates the environment
- initializes the policy network and optimizer
- starts the selected training algorithm
"""

import gymnasium as gym
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import config

from src.networks import PolicyNet
from src.reinforce import train_reinforce, run_episode
from src.utils import set_seed




def main():

    set_seed(config.SEED)

    env = gym.make(config.ENV_NAME, max_episode_steps=config.MAX_STEPS)
    #env.unwrapped.theta_threshold_radians = 0.418

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(
        state_dim,
        action_dim,
        config.HIDDEN_DIM
    )

    optimizer = optim.Adam(
        policy.parameters(),
        lr=config.LEARNING_RATE 
    )

    episode_rewards, eval_rewards, eval_lengths = train_reinforce(
        env=env,
        policy=policy,
        optimizer=optimizer,
        episodes=config.EPISODES,
        max_steps=config.MAX_STEPS,
        gamma=config.GAMMA,
        checkpoint_every=config.CHECKPOINT_EVERY,
        checkpoint_dir="lab3/checkpoints",
        N=config.N,
        M=config.M,
    )



    ### plot delle valutazioni periodiche dell'agente durante l'addestramento
    eval_x = [i * config.N for i in range(len(eval_rewards))]
    plt.figure()
    plt.plot(eval_x, eval_rewards, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Avg Total Reward (M episodi)")
    plt.title("Evaluation Reward")
    plt.show()

    plt.figure()
    plt.plot(eval_x, eval_lengths, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Avg Episode Length (M episodi)")
    plt.title("Evaluation Length")
    plt.show()



    if config.RENDER:
        policy.load_state_dict(torch.load("lab3/checkpoints/best.pt"))
        render_env = gym.make(config.ENV_NAME, render_mode="human", max_episode_steps=config.MAX_STEPS)
        render_env.unwrapped.theta_threshold_radians = 0.418 # 24 gradi al posto di 12(0.209) che era prima di default, questo lo faccio solo in testing perche in training tengo 12 che implica imparare in un ambiente piu difficile
        policy.eval()
        for _ in range(config.RENDER_EPISODES):
            run_episode(render_env, policy, max_steps=config.MAX_STEPS)
        render_env.close()
    env.close()


if __name__ == "__main__":
    main()