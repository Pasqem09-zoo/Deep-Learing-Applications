"""
REINFORCE implementation with optional learned value baseline.

Main functions:
- run_episode: runs one episode, collecting states, log-probabilities, rewards,
  total reward, and episode length.
- update_policy: updates the policy using discounted returns. If a value network
  is provided, it also trains the learned baseline V(s).
- evaluate_policy: runs several evaluation episodes and returns average reward
  and average episode length.
- train_reinforce: main training loop for REINFORCE, including logging,
  evaluation, and checkpoint saving.
"""





import os
import torch
import torch.nn.functional as F

from src.utils import select_action, compute_returns


def run_episode(env, policy, max_steps):
    states = []
    log_probs = []
    rewards = []

    state, info = env.reset()

    for step in range(max_steps):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action, log_prob = select_action(policy, state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        states.append(state_tensor)
        log_probs.append(log_prob)
        rewards.append(reward)

        state = next_state

        if done:
            break

    episode_reward = sum(rewards)
    episode_length = len(rewards)

    return torch.stack(states), log_probs, rewards, episode_reward, episode_length


def update_policy(optimizer, log_probs, rewards, gamma, states=None, value_net=None, value_optimizer=None):
    returns = compute_returns(rewards, gamma)
    returns = torch.tensor(returns, dtype=torch.float32)

    if value_net is not None and value_optimizer is not None and states is not None:
        values = value_net(states).squeeze(-1)
        advantages = returns - values.detach()
        policy_loss = -(torch.stack(log_probs) * advantages).sum()
        value_loss = F.mse_loss(values, returns)

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        return policy_loss.item(), value_loss.item()

    policy_loss = []
    for log_prob, G in zip(log_probs, returns):
        policy_loss.append(-log_prob * G)

    policy_loss = torch.stack(policy_loss).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    return policy_loss.item(), None


# Evaluate the current policy over multiple episodes
def evaluate_policy(env, policy, max_steps, eval_episodes):
    total_reward = 0.0
    total_length = 0

    policy.eval()
    with torch.no_grad():
        for _ in range(eval_episodes):
            _, _, _, episode_reward, episode_length = run_episode(
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
    eval_env,
    policy,
    optimizer,
    episodes,
    max_steps,
    gamma,
    checkpoint_every,
    checkpoint_dir,      # directory where model checkpoints are saved
    checkpoint_prefix,   # prefix used for checkpoint file names
    N,                   # evaluate every N episodes
    M,                   # number of evaluation episodes
    log_callback=None,
    value_net=None,
    value_optimizer=None,
):
    episode_rewards = []
    eval_rewards = []
    eval_lengths = []
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_reward = float('-inf')

    for episode in range(episodes):
        states, log_probs, rewards, episode_reward, episode_length = run_episode(
            env=env,
            policy=policy,
            max_steps=max_steps
        )

        policy_loss, value_loss = update_policy(
            optimizer=optimizer,
            log_probs=log_probs,
            rewards=rewards,
            gamma=gamma,
            states=states,
            value_net=value_net,
            value_optimizer=value_optimizer,
        )

        episode_rewards.append(episode_reward)

        if log_callback:
            log_data = {
                "train_reward": episode_reward,
                "reinforce_policy_loss": policy_loss,
            }
            if value_loss is not None:
                log_data["reinforce_value_loss"] = value_loss
            log_callback(log_data)

        if episode % 10 == 0:
            print(
                f"Episode {episode} | "
                f"Reward: {episode_reward:.1f} | "
                f"Length: {episode_length} | "
                f"Loss: {policy_loss:.3f}"
            )

        if episode % N == 0:
            avg_reward, avg_length = evaluate_policy(
                env=eval_env,
                policy=policy,
                max_steps=max_steps,
                eval_episodes=M
            )
            eval_rewards.append(avg_reward)
            eval_lengths.append(avg_length)
            if log_callback:
                log_callback({
                    "eval_avg_reward": avg_reward,
                    "eval_avg_length": avg_length,
                })
            print(
                f"Eval {episode} | "
                f"Avg Reward: {avg_reward:.1f} | "
                f"Avg Length: {avg_length:.1f}"
            )
        
        if episode % checkpoint_every == 0:
            torch.save(
                policy.state_dict(),
                os.path.join(checkpoint_dir, f"{checkpoint_prefix}_ep_{episode}.pt")
            )

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(
                policy.state_dict(),
                os.path.join(checkpoint_dir, f"{checkpoint_prefix}_best.pt")
            )

    return episode_rewards, eval_rewards, eval_lengths