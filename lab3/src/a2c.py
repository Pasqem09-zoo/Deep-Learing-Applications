"""
Advantage Actor-Critic (A2C) trainer.

This implementation is adapted from a Gymnasium tutorial structure
and extended for this project.

Main class:
- A2CTrainer: manages the full A2C training process, including interaction
  with the environment, actor-critic updates, evaluation, and checkpoint saving.

Main methods:
- run_n_steps: collects an n-step rollout from the current policy.
- compute_n_step_returns: computes bootstrapped discounted returns.
- evaluate_policy: evaluates the actor using greedy actions.
- train: runs the full A2C training loop.
"""





import os
import torch
import torch.nn.functional as F

from src.utils import select_action, select_greedy_action



class A2CTrainer:
	def __init__(self, env, actor, critic, actor_optim, critic_optim, config, checkpoint_dir, checkpoint_prefix, log_callback=None, eval_env=None):
		self.env = env
		self.eval_env = eval_env if eval_env is not None else env
		self.actor = actor
		self.critic = critic
		self.actor_optim = actor_optim
		self.critic_optim = critic_optim
		self.config = config
		self.checkpoint_dir = checkpoint_dir
		self.checkpoint_prefix = checkpoint_prefix
		self.log_callback = log_callback

	def run_n_steps(self, state):
		states = []
		log_probs = []
		entropies = []
		rewards = []
		done = False

		step_count = 0
		while step_count < self.config.A2C_N_STEPS and not done and step_count < self.config.MAX_STEPS:
			action, log_prob, dist = select_action(self.actor, state, return_dist=True)
			next_state, reward, terminated, truncated, _ = self.env.step(action)
			done = terminated or truncated

			states.append(torch.tensor(state, dtype=torch.float32))
			log_probs.append(log_prob)
			entropies.append(dist.entropy())
			rewards.append(reward)

			state = next_state
			step_count += 1

		return states, log_probs, entropies, rewards, state, done

	def evaluate_policy(self):
		total_reward = 0.0
		total_length = 0
		eval_env = self.eval_env  # use a separate evaluation environment

		self.actor.eval()
		with torch.no_grad():
			for _ in range(self.config.M):
				state, _ = eval_env.reset()
				done = False
				steps = 0
				while not done and steps < self.config.MAX_STEPS:
					action = select_greedy_action(self.actor, state)
					state, reward, terminated, truncated, _ = eval_env.step(action)
					done = terminated or truncated
					total_reward += reward
					steps += 1
					total_length += 1
		self.actor.train()

		avg_reward = total_reward / self.config.M
		avg_length = total_length / self.config.M
		return avg_reward, avg_length

	def compute_n_step_returns(self, rewards, next_value, done):
		returns = []
		R = 0.0 if done else next_value
		for r in reversed(rewards):
			R = r + self.config.GAMMA * R
			returns.insert(0, R)
		return returns

	def train(self):
		episode_rewards = []
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		best_reward = float("-inf")

		for episode in range(self.config.EPISODES):
			state, _ = self.env.reset()
			done = False
			episode_reward = 0.0
			episode_length = 0
			last_actor_loss = None
			last_critic_loss = None

			while not done:
				states, log_probs, entropies, rewards, next_state, done = self.run_n_steps(state)
				episode_reward += sum(rewards)
				episode_length += len(rewards)

				with torch.no_grad():
					next_value = self.critic(torch.tensor(next_state, dtype=torch.float32)).item()

				returns = self.compute_n_step_returns(rewards, next_value, done)

				states_tensor = torch.stack(states)
				returns_tensor = torch.tensor(returns, dtype=torch.float32)
				values = self.critic(states_tensor).squeeze(-1)
				advantages = returns_tensor - values.detach()
				
				# Normalize advantages within the rollout
				if self.config.A2C_ADV_NORM:
					if advantages.numel() > 1:  # avoid normalization when only one advantage is available (standard deviation would be zero)
						adv_mean = advantages.mean()
						adv_std = advantages.std(unbiased=False)
						advantages = (advantages - adv_mean) / (adv_std + 1e-8)
				
				entropy_mean = torch.stack(entropies).mean()

				actor_loss = -(torch.stack(log_probs) * advantages).mean()
				actor_loss -= self.config.A2C_ENTROPY_BETA * entropy_mean
				critic_loss = F.mse_loss(values, returns_tensor)
				last_actor_loss = actor_loss.item()
				last_critic_loss = critic_loss.item()

				self.actor_optim.zero_grad()
				actor_loss.backward()
				if self.config.A2C_GRAD_CLIP > 0:
					torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.A2C_GRAD_CLIP)
				self.actor_optim.step()

				self.critic_optim.zero_grad()
				critic_loss.backward()
				if self.config.A2C_GRAD_CLIP > 0:
					torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.A2C_GRAD_CLIP)
				self.critic_optim.step()

				state = next_state

			episode_rewards.append(episode_reward)

			if self.log_callback:
				self.log_callback({
					"train_reward": episode_reward,
					"actor_loss": last_actor_loss,
					"critic_loss": last_critic_loss,
				})

			if episode % self.config.N == 0:
				avg_reward, avg_length = self.evaluate_policy()

				if self.log_callback:
					self.log_callback({
						"eval_avg_reward": avg_reward,
						"eval_avg_length": avg_length,
					})
				print(
					f"Eval {episode} | Avg Reward: {avg_reward:.1f} | Avg Length: {avg_length:.1f}"
				)

				if avg_reward > best_reward:
					best_reward = avg_reward
					torch.save(
						self.actor.state_dict(),
						os.path.join(
							self.checkpoint_dir,
							f"{self.checkpoint_prefix}_best.pt"
						),
					)

			if episode % self.config.CHECKPOINT_EVERY == 0:
				torch.save(
					self.actor.state_dict(),
					os.path.join(self.checkpoint_dir, f"{self.checkpoint_prefix}_ep_{episode}.pt"),
				)

			if episode % 10 == 0:
				print(f"Episode {episode} | Reward: {episode_reward:.1f}")

		return episode_rewards
