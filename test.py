from src.envs.gridworld import GridWorld

env = GridWorld(width=5, height=5, n_coop=2, n_adv=1, n_targets=1, max_steps=10)
obs = env.reset()
print("Initial Observation:", obs)
env.render()

for t in range(5):
    actions = [0, 3]  # coop agent 0 moves up, agent 1 moves right
    adv_actions = [1] # adversary moves down
    obs, rewards, done = env.step(actions, adv_actions)
    print(f"Step {t+1} -> Obs: {obs}, Rewards: {rewards}, Done: {done}")
    env.render()
    if done:
        break
