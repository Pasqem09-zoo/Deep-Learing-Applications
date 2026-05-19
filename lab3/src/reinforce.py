'''
Reinforce algorithm implementation for training a policy in a reinforcement learning environment.
This file contains:
- run_episode: Executes a single episode in the environment, collecting log probabilities and rewards.
- update_policy: Updates the policy network using the collected log probabilities and rewards.
- train_reinforce: Main training loop that runs multiple episodes and updates the policy network.
'''






import os
import torch

from src.utils import select_action, compute_returns


def run_episode(env, policy, max_steps):
    log_probs = []
    rewards = []

    state, info = env.reset()

    for step in range(max_steps):
        action, log_prob = select_action(policy, state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(reward)

        state = next_state

        if done:
            break

    episode_reward = sum(rewards)
    episode_length = len(rewards)

    return log_probs, rewards, episode_reward, episode_length


def update_policy(optimizer, log_probs, rewards, gamma):
    returns = compute_returns(rewards, gamma)
    returns = torch.tensor(returns, dtype=torch.float32)

    policy_loss = []

    for log_prob, G in zip(log_probs, returns):
        policy_loss.append(-log_prob * G)

    policy_loss = torch.stack(policy_loss).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return policy_loss.item()


### serve per valutare ogni N iterazioni l'agente che runna M episodi nell'ambiente. 
### Questo è utile per monitorare i progressi dell'agente durante l'addestramento, e per salvare checkpoint periodici del modello.
def evaluate_policy(env, policy, max_steps, eval_episodes):
    total_reward = 0.0
    total_length = 0

    policy.eval()
    with torch.no_grad():
        for _ in range(eval_episodes):
            _, _, episode_reward, episode_length = run_episode(
                env=env,
                policy=policy,
                max_steps=max_steps
            )
            total_reward += episode_reward
            total_length += episode_length
    policy.train()

    avg_reward = total_reward / eval_episodes
    avg_length = total_length / eval_episodes

    return avg_reward, avg_length


def train_reinforce(
    env,
    policy,
    optimizer,
    episodes,
    max_steps,
    gamma,
    checkpoint_every,
    checkpoint_dir,
    N, ### ogni N episodi faccio una valutazione dell'agente
    M
):
    episode_rewards = []
    eval_rewards = []
    eval_lengths = []
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_reward = float('-inf')

    for episode in range(episodes):
        log_probs, rewards, episode_reward, episode_length = run_episode(
            env=env,
            policy=policy,
            max_steps=max_steps
        )

        policy_loss = update_policy(
            optimizer=optimizer,
            log_probs=log_probs,
            rewards=rewards,
            gamma=gamma
        )

        episode_rewards.append(episode_reward)

        if episode % 10 == 0:
            print(
                f"Episode {episode} | "
                f"Reward: {episode_reward:.1f} | "
                f"Length: {episode_length} | "
                f"Loss: {policy_loss:.3f}"
            )

        if episode % N == 0:
            avg_reward, avg_length = evaluate_policy(
                env=env,
                policy=policy,
                max_steps=max_steps,
                eval_episodes=M
            )
            eval_rewards.append(avg_reward)
            eval_lengths.append(avg_length)
            print(
                f"Eval {episode} | "
                f"Avg Reward: {avg_reward:.1f} | "
                f"Avg Length: {avg_length:.1f}"
            )
        
        if episode % checkpoint_every == 0:
            torch.save(
                policy.state_dict(),
                os.path.join(checkpoint_dir, f"periodic_ep_{episode}.pt")
            )

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(
                policy.state_dict(),
                os.path.join(checkpoint_dir, "best.pt")
            )

    return episode_rewards, eval_rewards, eval_lengths