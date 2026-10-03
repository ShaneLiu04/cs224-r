"""
Parallel seed sweep — runs 3 seeds for each training job on separate GPU containers.

Usage:
  modal run modal_train_para.py --algo iql --env-name antmaze-umaze-v0 --exp-name iql_run
  modal run --detach modal_train_para.py --algo awac --env-name antmaze-umaze-v0 --exp-name awac_run --use-wandb
  modal run modal_train_para.py --algo iql --env-name PointmassHard-v0 --exp-name iql_pm \
    --offline-dataset offline_datasets/pointmass_stitching_dataset.npz --use-wandb
"""

from modal_config import app, train

SEEDS = [1, 2, 3]


@app.local_entrypoint()
def main(
    algo: str = "iql",
    env_name: str = "antmaze-umaze-v0",
    exp_name: str = "modal_run",
    offline_dataset: str = "",
    filter_top_percent: float = 0.0,
    rew_shift: float = 0.0,
    rew_scale: float = 1.0,
    awac_lambda: float = 0.1,
    iql_expectile: float = 0.9,
    n_layers: int = 4,
    size: int = 512,
    learning_rate: float = 1e-4,
    num_timesteps: int = 0,
    batch_size: int = 256,
    use_wandb: bool = False,
    scalar_log_freq: int = 0,
    checkpoint_interval: int = -1,
    save_params: bool = False,
):
    """Dispatch one GPU container per seed, all running in parallel."""
    def build_extra_args(seed):
        extra = ["--seed", str(seed)]
        if offline_dataset:
            extra += ["--offline_dataset", offline_dataset]
        if filter_top_percent:
            extra += ["--filter_top_percent", str(filter_top_percent)]
        if rew_shift:
            extra += ["--rew_shift", str(rew_shift)]
        if rew_scale != 1.0:
            extra += ["--rew_scale", str(rew_scale)]
        if awac_lambda != 0.1:
            extra += ["--awac_lambda", str(awac_lambda)]
        if iql_expectile != 0.9:
            extra += ["--iql_expectile", str(iql_expectile)]
        if n_layers != 4:
            extra += ["--n_layers", str(n_layers)]
        if size != 512:
            extra += ["--size", str(size)]
        if learning_rate != 1e-4:
            extra += ["--learning_rate", str(learning_rate)]
        if num_timesteps:
            extra += ["--num_timesteps", str(num_timesteps)]
        if batch_size != 256:
            extra += ["--batch_size", str(batch_size)]
        if use_wandb:
            extra += ["--use_wandb"]
        if scalar_log_freq:
            extra += ["--scalar_log_freq", str(scalar_log_freq)]
        if checkpoint_interval != -1:
            extra += ["--checkpoint_interval", str(checkpoint_interval)]
        if save_params:
            extra += ["--save_params"]
        return extra

    configs = [
        (algo, env_name, f"{exp_name}_seed{seed}", build_extra_args(seed))
        for seed in SEEDS
    ]

    print(f"Launching {len(configs)} parallel jobs (seeds {SEEDS})...")
    for _ in train.starmap(configs):
        pass
