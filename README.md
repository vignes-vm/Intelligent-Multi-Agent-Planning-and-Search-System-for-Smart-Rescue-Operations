## Intelligent Multi-Agent Planning and Search for Smart Rescue Operations

This repository implements a multi-agent reinforcement learning framework for cooperative and adversarial search-and-rescue style gridworld tasks. The training pipeline is based on discrete-action Soft Actor-Critic (SAC) with centralized critics, decentralized policies, optional intrinsic motivation, and support for heterogeneous agent roles (cooperative and rogue/adversarial agents).

The system is designed for experiments in:

- Coordinated exploration in partially constrained environments
- Reward shaping with intrinsic novelty signals
- Communication-aware multi-agent behavior
- Mixed cooperative-adversarial rescue dynamics

## Highlights

- Multi-agent SAC training with centralized critic and decentralized policies
- Intrinsic reward modes based on count-based novelty
- Parallel environment rollouts for faster data collection
- Configurable map layouts and task setups
- Optional adversarial (rogue) agents
- Built-in logging and model checkpointing
- Rollout trajectory and reward export for analysis

## Repository Layout

```text
.
|-- timer.py                  # Main training/inference entry point
|-- algorithms/               # SAC variants
|   |-- sac.py
|   |-- sac_adv.py
|   |-- sac_hst.py
|-- envs/
|   |-- magw/
|   |   |-- multiagent_env.py # Gridworld environment
|   |   |-- load_env.py       # Search-and-rescue environment loader/utilities
|   |   |-- comms.py          # Communication/intrinsic reward helpers
|   |   |-- maps/             # Grid map definitions
|-- utils/                    # Replay buffer, wrappers, policies, critics, misc
|-- figs/                     # Figures/output artifacts (if generated)
```

## Requirements

No dependency lock file is currently included. Based on imports in the codebase, install at least:

- Python 3.8+
- PyTorch
- NumPy
- Gym
- tensorboardX
- pygame
- matplotlib
- pandas

Example installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy gym tensorboardX pygame matplotlib pandas
```

## Quick Start

Run a baseline gridworld training job:

```bash
python timer.py baseline_run
```

Run with custom map and agent/object counts:

```bash
python timer.py exp_map21 \
	--env_type gridworld \
	--map_ind 21 \
	--num_agents 2 \
	--num_objects 2 \
	--n_rollout_threads 12 \
	--train_time 10000
```

Run with rogue agents (example: agent 3 is adversarial):

```bash
python timer.py exp_rogue \
	--num_agents 3 \
	--num_objects 2 \
	--rogue_agents 3 \
	--rogue_reward_factor 1.0
```

Run with model loading/inference-oriented mode:

```bash
python timer.py eval_run \
	--load_model True \
	--model_path /absolute/path/to/model.pt \
	--inference 1
```

## Key Training Arguments

`timer.py` exposes many CLI options. Commonly tuned parameters include:

- `model_name`: Output experiment name (required positional argument)
- `--map_ind`: Grid map index
- `--num_agents`: Number of agents
- `--num_objects`: Number of targets/treasures
- `--train_time`: Total training timesteps
- `--n_rollout_threads`: Parallel rollout environments
- `--batch_size`: SAC batch size
- `--num_updates`: Number of update steps per cycle
- `--intrinsic_reward`: Enable/disable intrinsic rewards (`0` or `1`)
- `--explr_types`: Intrinsic exploration heads (`0-4`)
- `--beta`: Intrinsic reward weighting
- `--output_dir`: Directory for rollout/reward exports
- `--load_model`, `--model_path`, `--inference`: Evaluation/loading controls

To view the full argument list, run:

```bash
python timer.py -h
```

## Outputs and Artifacts

During training, the project writes:

- Model checkpoints under `models/<env_type>/<map_descriptor>/<model_name>/runX/`
- TensorBoard logs in each run's `logs/` folder
- Incremental model snapshots in `incremental/`
- Rollout exports under `--output_dir` (default: `out_baseline`), including:
	- `global_time.txt`
	- `rewards_with_time.csv`
	- `adv_rewards_with_time.csv`
	- rollout CSV/figure folders (when visualization/export conditions are met)

Launch TensorBoard:

```bash
tensorboard --logdir models
```

## Environment Notes

- Map files are loaded from `envs/magw/maps/` using the naming pattern `map<index>_<num_objects>_multi.txt`.
- The default environment type is `gridworld`.
- Communication radius and stochastic target behavior can be configured through CLI flags such as `--comm_radius`, `--random_target`, `--random_epsilon`, and `--number_of_random_targets`.

## Reproducibility Tips

- Keep `model_name` unique per run to avoid confusion between experiment folders.
- Fix map index, agent count, object count, and rollout thread count when comparing algorithms.
- Log all command-line arguments used for each experiment.

## Troubleshooting

- If `pygame` fails in headless environments, ensure SDL headless mode is available (the environment code sets `SDL_VIDEODRIVER=dummy`).
- If memory usage is high, reduce:
	- `--buffer_length`
	- `--n_rollout_threads`
	- `--batch_size`
- If training stalls, try shorter `--max_episode_length` and verify map/task settings.

## Future Improvements

- Add a pinned `requirements.txt` for reproducible setup
- Add scripts for standardized training/evaluation runs
- Add tests for environment loading and reward computation

## License

No license file is currently included in this repository. Add a `LICENSE` file before external distribution.

