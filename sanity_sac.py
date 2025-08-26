import numpy as np
from src.marl.replay_buffer import MultiAgentReplayBuffer
from src.marl.sac import MultiAgentSAC

N = 2             # two cooperative agents
obs_dim = 4       # toy obs vector length per agent (we'll wire real encoding next step)
A = 4             # up/down/left/right

sac = MultiAgentSAC(n_agents=N, obs_dim_per_agent=obs_dim, n_actions=A, device="cpu")

buf = MultiAgentReplayBuffer(capacity=1000)

# Populate with random toy data
for _ in range(64):
    obs = [np.random.randn(obs_dim).astype(np.float32) for _ in range(N)]
    actions = [np.random.randint(0, A) for _ in range(N)]
    rewards = [np.random.randn() for _ in range(N)]
    next_obs = [np.random.randn(obs_dim).astype(np.float32) for _ in range(N)]
    done = np.random.rand() < 0.1
    buf.push(obs, actions, rewards, next_obs, done)

batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done = buf.sample(32)
log = sac.update(batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done)
print("Update log:", log)

# Try selecting actions from dummy obs
test_obs = [np.zeros(obs_dim, dtype=np.float32) for _ in range(N)]
print("Greedy actions:", sac.select_actions(test_obs, eval_mode=True))
print("Stochastic actions:", sac.select_actions(test_obs, eval_mode=False))
