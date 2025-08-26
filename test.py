from src.envs.gridworld import GridWorld
from src.agents.cooperative import CooperativeAgent
from src.agents.adversarial import AdversarialAgent

# Create environment
env = GridWorld(width=5, height=5, n_coop=2, n_adv=1, n_targets=1, max_steps=10)
obs = env.reset()

# Create agents
coop_agents = [CooperativeAgent(i) for i in range(2)]
adv_agents = [AdversarialAgent(i) for i in range(1)]

env.render()

for t in range(5):
    coop_actions = [agent.act(obs) for agent in coop_agents]
    adv_actions = [agent.act(obs) for agent in adv_agents]
    obs, rewards, done = env.step(coop_actions, adv_actions)
    print(f"Step {t+1}: Coop actions={coop_actions}, Adv actions={adv_actions}, Rewards={rewards}, Done={done}")
    env.render()
    if done:
        break
