from modal_config import app, train


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
    seed: int = 1,
    use_wandb: bool = False,
    scalar_log_freq: int = 0,
    checkpoint_interval: int = -1,
    save_params: bool = False,
):
    extra_args = ["--seed", str(seed)]
    if offline_dataset:
        extra_args += ["--offline_dataset", offline_dataset]
    if filter_top_percent:
        extra_args += ["--filter_top_percent", str(filter_top_percent)]
    if rew_shift:
        extra_args += ["--rew_shift", str(rew_shift)]
    if rew_scale != 1.0:
        extra_args += ["--rew_scale", str(rew_scale)]
    if awac_lambda != 0.1:
        extra_args += ["--awac_lambda", str(awac_lambda)]
    if iql_expectile != 0.9:
        extra_args += ["--iql_expectile", str(iql_expectile)]
    if n_layers != 4:
        extra_args += ["--n_layers", str(n_layers)]
    if size != 512:
        extra_args += ["--size", str(size)]
    if learning_rate != 1e-4:
        extra_args += ["--learning_rate", str(learning_rate)]
    if num_timesteps:
        extra_args += ["--num_timesteps", str(num_timesteps)]
    if batch_size != 256:
        extra_args += ["--batch_size", str(batch_size)]
    if use_wandb:
        extra_args += ["--use_wandb"]
    if scalar_log_freq:
        extra_args += ["--scalar_log_freq", str(scalar_log_freq)]
    if checkpoint_interval != -1:
        extra_args += ["--checkpoint_interval", str(checkpoint_interval)]
    if save_params:
        extra_args += ["--save_params"]

    train.remote(algo, env_name, f"{exp_name}_seed{seed}", extra_args)
