# 环境与工具

## 按作业的环境需求

| 作业 | Python | 关键依赖 | 计算 |
| --- | --- | --- | --- |
| HW1（Flappy Bird） | 3.10+ | PyTorch、Gymnasium 风格自建 env（`pip install -e .`） | CPU 可跑 |
| HW2（hammer） | conda env（`conda_env_modal.yml`） | PyTorch、Meta-World、MuJoCo、Hydra、W&B、Modal | Modal |
| HW3（AntMaze） | 3.10.19 | PyTorch、D4RL、MuJoCo、W&B、Modal | Modal |
| Default Project | 3.10+ | transformers、vLLM、Ray、W&B、Modal | 单卡/Modal H100 |

## 基础工具链

- **PyTorch 2.x**：课程作业统一框架；
- **Gymnasium / MuJoCo**：连续控制环境；
- **D4RL**：离线数据集（首次运行自动下载到 `~/.d4rl`）；
- **Meta-World**：HW2 的 `hammer-v2` 任务；
- **Hydra**：配置文件管理（HW2）；
- **Weights & Biases**：实验追踪（HW2/HW3/项目）；
- **Modal**：官方计算平台（HW2/HW3/项目）。

## Modal 替代方案

课程不保证非 Modal 环境，但可以本地/自建替代：

1. **本地 GPU（Linux + NVIDIA）**：HW2 提供了 `conda_env_local.yml`（官方不提供支持）；
2. **自建云 GPU**：把 `modal_*.py` 中的镜像/卷指令映射为普通训练脚本；
   D4RL 数据可在本地下载；
3. **降低规模**：减少并行环境数、训练步数、模型大小，先验证正确性；
4. **无 GPU**：HW1 与论文精读、网格世界、部分 analysis 可完成；HW2/HW3 训练需 GPU。

## W&B 离线模式

```bash
export WANDB_MODE=offline     # 本地调试
wandb sync wandb/offline-run-*   # 有网后再同步
```

## Default Project 的资源

- 模型：`Qwen/Qwen2.5-0.5B`（默认），需要 tokenizer 与 bf16 支持；
- 采样：vLLM + Ray；训练与采样可分离（参考 `rloo_trainer/`）；
- 存储：HF Hub（上传 checkpoint）+ W&B（指标）。

## 常用命令

```bash
# 环境
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# 实验
python main.py --method bc_reg --env hard
modal run --detach modal_train_para.py --algo iql --env-name antmaze-umaze-v0 ...

# 数据/结果
modal volume ls cs224r-hw3-results
modal volume get cs224r-hw3-results / ./data/
```

## 常见坑

- Windows 上 MuJoCo/D4RL 支持有限，建议 Linux 环境或 WSL；
- MuJoCo 版本与 gym 版本的兼容矩阵是历史作业最大的坑（2023 版尤其）；
- vLLM 与 PyTorch/CUDA 版本强绑定，按项目 `pyproject.toml` 安装；
- W&B 在 Modal 中需要 `modal secret create wandb-secret ...`。
