"""
Reinforcement Learning Agents for Crypto Trading:
- Proximal Policy Optimization (PPO) with Actor-Critic Architecture
- Experience Rollout Buffer with Generalized Advantage Estimation (GAE)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Orthogonal layer initialization for stable RL training."""
    if isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network with shared representations and dedicated policy/value heads.
    """
    def __init__(self, state_dim: int = 31, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
        )
        
        # Actor head (Policy logits)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 64)),
            nn.LeakyReLU(0.1),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        
        # Critic head (State-Value V(s))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 64)),
            nn.LeakyReLU(0.1),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(state)
        return self.critic(features)

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state)
        logits = self.actor(features)
        dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(features)
        
        return action, log_prob, entropy, value


class RolloutBuffer:
    """Stores trajectories collected during rollouts for PPO updates."""
    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.reset()
        
    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.ptr = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates Generalized Advantage Estimation (GAE)."""
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_val = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_val = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
                
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        
        # Convert to torch tensors
        b_states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        b_actions = torch.tensor(np.array(self.actions), dtype=torch.long, device=self.device)
        b_log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device)
        b_advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        b_returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        b_values = torch.tensor(np.array(self.values), dtype=torch.float32, device=self.device)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        return b_states, b_actions, b_log_probs, b_advantages, b_returns, b_values


class PPOAgent:
    """
    PPO Agent with Actor-Critic policy gradient updates.
    """
    def __init__(
        self,
        state_dim: int = 31,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.20,
        entropy_coef: float = 0.015,
        value_loss_coef: float = 0.50,
        max_grad_norm: float = 0.50,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.actor_critic = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(capacity=2000, state_dim=state_dim, device=device)

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            if deterministic:
                features = self.actor_critic.feature_extractor(state_tensor)
                logits = self.actor_critic.actor(features)
                action = torch.argmax(logits, dim=-1)
                log_prob = torch.tensor([0.0], device=self.device)
                value = self.actor_critic.critic(features)
            else:
                action, log_prob, _, value = self.actor_critic.get_action_and_value(state_tensor)

        return action.item(), log_prob.item(), value.item()

    def update(
        self,
        last_value: float,
        batch_size: int = 64,
        mini_epochs: int = 4,
    ) -> Dict[str, float]:
        """Performs PPO policy and value network updates."""
        (
            b_states,
            b_actions,
            b_old_log_probs,
            b_advantages,
            b_returns,
            b_values,
        ) = self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        n_samples = len(b_states)
        indices = np.arange(n_samples)
        
        pg_losses, v_losses, ent_losses = [], [], []

        for _ in range(mini_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                _, new_log_probs, entropy, new_values = self.actor_critic.get_action_and_value(
                    b_states[batch_idx], b_actions[batch_idx]
                )

                # Policy Loss with PPO-Clip
                log_ratio = new_log_probs - b_old_log_probs[batch_idx]
                ratio = torch.exp(log_ratio)

                mb_adv = b_advantages[batch_idx]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value Loss with clipping
                v_loss = 0.5 * ((new_values.squeeze(-1) - b_returns[batch_idx]) ** 2).mean()

                # Entropy Bonus
                entropy_loss = entropy.mean()

                # Total Loss
                total_loss = pg_loss + self.value_loss_coef * v_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy_loss.item())

        self.buffer.reset()

        return {
            "policy_loss": float(np.mean(pg_losses)) if pg_losses else 0.0,
            "value_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "entropy": float(np.mean(ent_losses)) if ent_losses else 0.0,
        }
