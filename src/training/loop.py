import numpy as np
from src.envs.gridworld import GridWorld
from src.marl.replay_buffer import MultiAgentReplayBuffer
from src.marl.sac import MultiAgentSAC

# === OBS ENCODER ===
def encode_obs(obs_dict, agent_idx):
    """
    Encode observation for one agent into numeric vector.
    Format: [x, y, dx_to_target, dy_to_target, steps_norm]
    """
    x, y = obs_dict["coop_agents"][agent_idx]
    if obs_dict["targets"]:
        tx, ty = obs_dict["targets"][0]
        dx, dy = tx - x, ty - y
    else:
        dx, dy = 0, 0
    steps_norm = obs_dict["steps"] / 100.0
    return np.array([x, y, dx, dy, steps_norm], dtype=np.float32)


def run_training(episodes=20, max_steps=50, batch_size=32):
    env = GridWorld(width=5, height=5, n_coop=2, n_adv=0, n_targets=1, max_steps=max_steps)

    obs_dim = 5
    A = 4
    N = 2

    sac = MultiAgentSAC(n_agents=N, obs_dim_per_agent=obs_dim, n_actions=A, device="cpu")
    buf = MultiAgentReplayBuffer(capacity=5000)

    for ep in range(episodes):
        obs = env.reset()
        obs_list = [encode_obs(obs, i) for i in range(N)]
        total_reward = 0
        success = False
        logs_accum = {"critic_loss": [], "actor_loss": [], "entropy": []}

        for t in range(max_steps):
            # Select actions
            actions = sac.select_actions(obs_list)
            # Step in environment
            next_obs, rewards, done = env.step(actions)
            next_obs_list = [encode_obs(next_obs, i) for i in range(N)]

            # Store experience
            buf.push(obs_list, actions, rewards, next_obs_list, done)

            # Update SAC
            if len(buf) > batch_size:
                batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done = buf.sample(batch_size)
                log = sac.update(batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_done)
                logs_accum["critic_loss"].append(log["critic_loss"])
                logs_accum["actor_loss"].append(log["actor_loss"])
                logs_accum["entropy"].append(log["avg_entropy"])

            obs_list = next_obs_list
            total_reward += sum(rewards)

            if done:
                if not next_obs["targets"]:  # all targets found
                    success = True
                break

        # Episode logs
        avg_closs = np.mean(logs_accum["critic_loss"]) if logs_accum["critic_loss"] else 0
        avg_aloss = np.mean(logs_accum["actor_loss"]) if logs_accum["actor_loss"] else 0
        avg_ent = np.mean(logs_accum["entropy"]) if logs_accum["entropy"] else 0
        print(f"Ep {ep+1}/{episodes} | Reward={total_reward:.1f} | Success={success} | "
              f"CriticLoss={avg_closs:.3f} | ActorLoss={avg_aloss:.3f} | Entropy={avg_ent:.3f}")

    print("Training loop finished ✅")
