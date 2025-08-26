import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mlp(sizes, activation=nn.ReLU, out_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_activation
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)


class DiscreteActor(nn.Module):
    """
    Categorical policy over discrete actions for a single agent.
    Input: agent observation vector
    Output: logits over actions (softmax -> probabilities)
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: Tuple[int, int] = (128, 128)):
        super().__init__()
        self.net = mlp([obs_dim, hidden[0], hidden[1], n_actions])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Returns logits (unnormalized)
        return self.net(obs)

    def action_and_logp(self, obs: torch.Tensor, temperature: float = 1.0):
        logits = self.forward(obs) / temperature
        probs = F.softmax(logits, dim=-1)
        # Categorical sample
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, probs, logits


class CentralizedCritic(nn.Module):
    """
    Q(s,a) networks with centralized state-action input.
    We use Double-Q trick: two critics and target copies.
    """
    def __init__(self, joint_obs_dim: int, joint_act_dim: int, hidden: Tuple[int, int] = (256, 256)):
        super().__init__()
        in_dim = joint_obs_dim + joint_act_dim
        self.q1 = mlp([in_dim, hidden[0], hidden[1], 1])
        self.q2 = mlp([in_dim, hidden[0], hidden[1], 1])

    def forward(self, joint_obs: torch.Tensor, joint_act_onehot: torch.Tensor):
        x = torch.cat([joint_obs, joint_act_onehot], dim=-1)
        return self.q1(x), self.q2(x)


def one_hot(actions: torch.Tensor, n_actions_per_agent: int) -> torch.Tensor:
    """
    actions: [B, N] tensor of ints
    returns: [B, N * n_actions_per_agent] one-hot
    """
    B, N = actions.shape
    out = torch.zeros(B, N * n_actions_per_agent, device=actions.device)
    for i in range(N):
        idx = actions[:, i] + i * n_actions_per_agent
        out[torch.arange(B), idx] = 1.0
    return out


class MultiAgentSAC:
    """
    Discrete SAC with:
      - Per-agent categorical actors π_i(a_i | o_i)
      - Centralized twin critics Q1, Q2 (see all agents' obs and actions)
    Assumptions:
      - Each agent has same discrete action space size (A)
      - Observations are pre-encoded into fixed-size vectors per agent
    """
    def __init__(
        self,
        n_agents: int,
        obs_dim_per_agent: int,
        n_actions: int,
        gamma: float = 0.99,
        tau: float = 0.01,
        alpha: float = 0.2,     # entropy temperature (can be learned later)
        lr: float = 3e-4,
        device: str = "cpu"
    ):
        self.N = n_agents
        self.obs_dim = obs_dim_per_agent
        self.A = n_actions
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = torch.device(device)

        # Actors (one per agent)
        self.actors = nn.ModuleList([
            DiscreteActor(self.obs_dim, self.A) for _ in range(self.N)
        ]).to(self.device)

        # Centralized critics (twin) + target copies
        joint_obs_dim = self.N * self.obs_dim
        joint_act_dim = self.N * self.A
        self.critic = CentralizedCritic(joint_obs_dim, joint_act_dim).to(self.device)
        self.target_critic = CentralizedCritic(joint_obs_dim, joint_act_dim).to(self.device)
        self._hard_update(self.target_critic, self.critic)

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actors.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def _hard_update(self, target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        with torch.no_grad():
            for p_t, p in zip(target.parameters(), source.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p.data)

    def select_actions(self, obs_list: List[np.ndarray], eval_mode: bool = False) -> List[int]:
        """
        obs_list: list of per-agent obs (np arrays) -> returns list of int actions
        """
        actions = []
        for i, actor in enumerate(self.actors):
            o = torch.as_tensor(obs_list[i], dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = actor(o)
            probs = F.softmax(logits, dim=-1)
            if eval_mode:
                a = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs=probs)
                a = dist.sample()
            actions.append(int(a.item()))
        return actions

    def _encode_obs_list(self, obs_list_batch: List[List[np.ndarray]]) -> torch.Tensor:
        """
        obs_list_batch: list (B) of obs_list (N) -> [B, N*obs_dim]
        """
        B = len(obs_list_batch)
        out = torch.zeros((B, self.N * self.obs_dim), dtype=torch.float32, device=self.device)
        for b, obs_list in enumerate(obs_list_batch):
            vecs = []
            for i in range(self.N):
                # Expect obs_list[i] already a numeric vector; if dict/tuple, user should encode upstream.
                vecs.append(np.asarray(obs_list[i], dtype=np.float32))
            out[b] = torch.from_numpy(np.concatenate(vecs, axis=-1)).to(self.device)
        return out

    def _policy_eval(self, obs_batch: torch.Tensor):
        """
        For each agent i, compute:
          - logits_i(o_i)
          - probs_i(o_i)
          - sampled a_i and logπ_i(a_i|o_i)
        Returns tensors stacked by agent.
        """
        B = obs_batch.shape[0]
        # Split obs per agent
        obs_agents = obs_batch.view(B, self.N, self.obs_dim)

        actions = []
        logps = []
        probs_list = []
        logits_list = []

        for i, actor in enumerate(self.actors):
            logits = actor(obs_agents[:, i, :])            # [B, A]
            probs = F.softmax(logits, dim=-1)              # [B, A]
            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample()                               # [B]
            logp = dist.log_prob(a)                        # [B]
            actions.append(a)
            logps.append(logp)
            probs_list.append(probs)
            logits_list.append(logits)

        actions = torch.stack(actions, dim=1)              # [B, N]
        logps = torch.stack(logps, dim=1)                  # [B, N]
        return actions, logps, probs_list, logits_list

    def update(self,
               batch_obs: List[List[np.ndarray]],
               batch_actions: List[List[int]],
               batch_rewards: List[List[float]],
               batch_next_obs: List[List[np.ndarray]],
               batch_dones: np.ndarray,
               reward_aggregation: str = "sum"):
        """
        One gradient update step for critics and actors.

        reward_aggregation:
          - "sum": cooperative team objective sums per-agent rewards
          - "mean": average per-agent rewards
          (Later we can add adversarial shaping etc.)
        """
        device = self.device

        # Encode joint obs
        obs_joint = self._encode_obs_list(batch_obs)               # [B, N*obs_dim]
        next_obs_joint = self._encode_obs_list(batch_next_obs)

        # Actions -> [B, N] tensor, then one-hot -> [B, N*A]
        acts = torch.as_tensor(np.array(batch_actions), dtype=torch.long, device=device)
        acts_onehot = one_hot(acts, self.A)

        # Rewards aggregate to team scalar for value bootstrap (cooperative baseline)
        rew = torch.as_tensor(np.array(batch_rewards), dtype=torch.float32, device=device)  # [B, N]
        if reward_aggregation == "sum":
            team_rew = rew.sum(dim=1, keepdim=True)     # [B,1]
        else:
            team_rew = rew.mean(dim=1, keepdim=True)

        done = torch.as_tensor(batch_dones.reshape(-1, 1), dtype=torch.float32, device=device)

        # -------- Target Value --------
        with torch.no_grad():
            # Next actions sampled from current policy (for SAC target)
            next_actions, next_logps, next_probs_list, _ = self._policy_eval(next_obs_joint)
            next_onehot = one_hot(next_actions, self.A)
            q1_t, q2_t = self.target_critic(next_obs_joint, next_onehot)
            q_t_min = torch.min(q1_t, q2_t)                         # [B,1]

            # Entropy term: sum of per-agent logπ(a_i|o_i)
            # next_logps: [B, N] -> team entropy contribution
            entropy_term = self.alpha * next_logps.sum(dim=1, keepdim=True)
            target = team_rew + (1.0 - done) * self.gamma * (q_t_min - entropy_term)

        # -------- Critic Update --------
        q1, q2 = self.critic(obs_joint, acts_onehot)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        # -------- Actor Update --------
        actions_new, logps_new, probs_new, logits_new = self._policy_eval(obs_joint)
        onehot_new = one_hot(actions_new, self.A)
        q1_pi, q2_pi = self.critic(obs_joint, onehot_new)
        q_pi_min = torch.min(q1_pi, q2_pi)  # [B,1]

        # Actor objective: maximize (Q - alpha * entropy)
        # Here entropy = sum_i logπ_i(a_i|o_i)
        entropy_now = logps_new.sum(dim=1, keepdim=True)
        actor_loss = (self.alpha * entropy_now - q_pi_min).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actors.parameters(), max_norm=1.0)
        self.actor_opt.step()

        # -------- Target Soft Update --------
        self._soft_update(self.target_critic, self.critic, self.tau)

        # For logging/inspection
        out = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "avg_entropy": float((-entropy_now).mean().item())  # negative logp ~ entropy proxy
        }
        return out
