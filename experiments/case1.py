from src.training.loop import run_training

if __name__ == "__main__":
    print("=== Case I: Cooperative Agents (Coverage Reward) ===")
    run_training(episodes=20, max_steps=50)
