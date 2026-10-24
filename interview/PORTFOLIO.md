# 项目作品集：面试用 STAR 故事

本页把仓库里的实现与实验整理成可以在面试中讲 2–3 分钟的故事。每条都按
Situation / Task / Action / Result 组织，数字全部来自 `solutions/` 下的原始指标与报告，
可以当场打开图和代码。后面的「追问」是面试官最可能接着问的问题。

作者：ShaneLiu04（[GitHub](https://github.com/ShaneLiu04)）

## 索引

| # | 主题 | 考察点 | 证据 |
| --- | --- | --- | --- |
| 1 | Countdown RLOO 停止符 bug | LLM RL 调试、tokenizer | [DP README](../solutions/spring2026/default-project/README.md) |
| 2 | 0.5B 模型 RL 显存与吞吐 | 系统、显存估算 | [ENGINEERING](ENGINEERING.md) |
| 3 | 用解析恒等式验证 IPO/DPO | 测试设计 | [DP tests](../solutions/spring2026/default-project/tests/) |
| 4 | 发现 CQL starter 的数据采集器是 oracle | 读代码、实验设计 | [2025 HW3](../solutions/spring2025/homework3-offline-rl/README.md) |
| 5 | RND 数据 vs 随机数据 | 离线 RL 数据覆盖 | [2025 HW3](../solutions/spring2025/homework3-offline-rl/README.md) |
| 6 | HER 让 25-bit 从 0 到 1 | 稀疏奖励、重标记 | [2025 HW4](../solutions/spring2025/homework4-goal-meta-rl/README.md) |
| 7 | DREAM vs RL²：探索与利用解耦 | meta-RL、信息增益 | [2025 HW4](../solutions/spring2025/homework4-goal-meta-rl/README.md) |
| 8 | 失效的上游依赖与数据集 | 工程韧性 | [2026 HW2](../solutions/spring2026/homework2-online-rl/README.md)、[2026 HW3](../solutions/spring2026/homework3-offline-rl/README.md) |
| 9 | 单卡云实验调度 | 资源利用、成本 | [ENGINEERING](ENGINEERING.md) |

---

## 1. Countdown RLOO：奖励在涨，样本却越过了答案标签

- **S**：Qwen2.5-0.5B 在 Countdown 上做 SFT → RLOO。第一批 RLOO 的样本经常在
  `</answer>` 之后继续生成，格式奖励与精确匹配的关系异常。
- **T**：在不改奖励函数的前提下找到生成越界的原因。
- **A**：逐 token 打印生成结果，发现 `</answer>` 的 BPE 切分取决于前面是否有空格，
  停止序列只登记了一种 token 编号。修正为同时接受两种切分，并在闭合标签处截断
  损失，避免对标签之后的 token 做策略梯度。
- **R**：修正后的生成在闭合标签处停止；但 120 步 RLOO 的 pass@8（0.340）仍未稳定超过
  SFT，据此判断短程小批量 RL 主要在调整格式而不是提升解题能力。后续在长 think SFT
  检查点上做的长程 RLOO 把留出集 pass@1 从 0.24 提到 0.505、pass@4 从 0.445 提到 0.60。
- **追问**：为什么不直接用字符串停止？（采样引擎按 token 判停；字符串停止要在 decode
  后回扫，批量时更贵且和损失掩码不一致。）pass@1 涨、pass@k 涨得少说明什么？
  （RL 主要在 sharpen 已有分布，见 Yue et al. 2025。）

## 2. 0.5B 模型 RL 的显存与吞吐

- **S**：单卡上需要同时容纳策略、参考模型、优化器状态和采样。
- **T**：让 RLOO 在不降低组大小的前提下跑满 GPU。
- **A**：词表维度的 log-sum-exp 用 float32 且按时间分块；训练态注意力换成 scaled
  dot-product attention，避免物化 `[B,H,T,T]`；评测用 vLLM，但发现它不能和 RL 训练
  并行——PyTorch caching allocator 在训练中保留约 89 GB，vLLM 启动时按
  `gpu_memory_utilization × 总显存` 申请，必然失败。改成单队列串行，评测把
  utilization 设为 0.3。
- **R**：长程运行稳定在 22–35 s/step，GPU 利用率接近 100%，没有 OOM 中断。
- **追问**：AdamW 下 0.5B 参数的显存账？（权重 bf16 1 GB + fp32 主权重 2 GB + 两个
  动量 4 GB + 梯度，约 8 GB，激活另算。）为什么 `torch.cuda.empty_cache()` 不能解决？
  （进程间不共享 allocator 缓存，释放时机也不可控。）

## 3. 用解析恒等式给偏好优化写单元测试

- **S**：IPO/DPO 训练曲线「看起来在下降」不代表实现正确。
- **T**：找到不依赖训练结果的正确性检查。
- **A**：策略等于参考策略时 `Δ_w = Δ_l = 0`，于是 IPO 损失恒为 `(1/(2β))²`，DPO 损失恒为
  `log 2`。把这两个恒等式写成单元测试，另加 SFT 掩码右移、提示不反传、RLOO 优势等测试。
- **R**：`β=0.1` 时第一步 IPO 损失正好是 25，DPO 是 0.693。随后发现求和版本的验证
  损失升到 31.07，而长度平均版本维持在 25.01，偏好准确率都停在 0.5，说明损失下降
  不能单独当作对齐完成的证据。
- **追问**：为什么序列求和的 IPO 不稳定？（长度不同的回答 `Δ` 量级不同，平方损失
  被长回答主导。）

## 4. CQL starter 的「探索数据」其实是 oracle 数据

- **S**：2025 HW3 要求比较 RND 探索与随机探索收集的数据对 CQL 的影响。
- **T**：解释为什么 `--use_rnd` 开关几乎不改变结果。
- **A**：读 `explore_or_exploit_agent.step_env`，发现无论开关如何都调用
  `env.get_optimal_action`——在全源最短路距离上贪心的一步前瞻，即专家。`--use_rnd`
  只改变 `learning_starts`。新增 `--behavior {oracle,explore}` 参数（默认保持原行为），
  并单独跑了 oracle 基线评估。
- **R**：oracle 数据下 α=0 的 CQL 全部种子在 Medium 上 −150（失败），α≥0.01 即可解，
  OOD gap 随 α 单调下降（1.91 → 0.12）。换成真实探索数据后结论改变（见故事 5）。
- **追问**：如何确认是 starter 而不是自己的 bug？（固定种子对比两个开关下的 replay
  buffer 内容与动作分布。）

## 5. 离线 RL 的数据覆盖：RND 数据 vs 随机数据

- **S**：同一个 Medium 迷宫，用 RND 策略或 ε-随机策略收集离线数据。
- **T**：判断「探索得更好」的数据是否总是更适合离线学习。
- **A**：IQL 与 CQL 各跑多种子，α ∈ {0, 0.1, 1}、ζ ∈ {0.5, 0.7, 0.9, 0.95}。
- **R**：IQL（ζ=0.7）RND 数据 −50.85±2.43 远好于随机数据 −90.69±6.72；CQL 反过来，
  随机数据 + α=0.1 得 −34.3±6.1（6 seeds），RND 数据 + α=1 全部失败（−150）。
  RND 数据覆盖广但每个状态的动作稀疏，强保守正则把策略钉在数据里很差的动作上。
- **追问**：为什么 IQL 不受这个影响？（expectile 只在数据动作上学 V 的上分位数，
  不显式压低 OOD 动作。）

## 6. HER：25-bit flip 从 0 到 1

- **S**：bit-flip 环境奖励 0/−1，目标随机，DQN 几乎永远拿不到正信号。
- **T**：实现 final / future / random 三种重标记并量化差异。
- **A**：每个 episode 额外写入 k 条用实际到达状态替换目标的 transition；
  DQN 目标裁剪到 `[−1/(1−γ), 0]`。
- **R**：无 hindsight 时 15/25 bit 成功率为 0，6 bit 0.896±0.143；所有 HER 变体最终都到
  1.0，25 bit 到 0.9 需 290–320 epoch。k ∈ {1,2,4,8} 在 15 bit 上没有差别。
  Sawyer 推物（稠密奖励、5 seeds）：无 hindsight 0.547±0.106，random 0.994±0.006。
- **追问**：HER 什么时候会有偏？（随机环境里重标记改变了目标条件下的转移分布。）

## 7. DREAM vs RL²：把探索和利用分开学

- **S**：网格 meta-RL，走路回报 0.1，读地图后坐公交可得最优 0.7。
- **T**：比较端到端 RL² 与 DREAM。
- **A**：DREAM 用解码器把探索轨迹编码为任务信息，探索奖励为信息增益减每步代价 c；
  做 c=0 消融。
- **R**：DREAM 0.696±0.007（5 seeds）接近最优，RL² 0.143±0.073。c=0 时回报不变（0.70），
  但探索长度涨到 9.9±0.16（接近上限），说明 c 控制的是探索效率而不是能否学会。
- **追问**：RL² 为什么失败？（探索收益延迟到下一个 episode，端到端信用分配太难。）

## 8. 失效的上游依赖与数据集

- **S**：2026 HW2 依赖的 MuJoCo 2.1 tarball 已下线；2026 HW3 公开的
  `antmaze-medium-diverse-v0` HDF5 被截断。
- **A**：HW2 用当前 `mujoco` + Meta-World，在 `mw.py` 兼容四元组/五元组 step API，
  observation、action、success 与 action repeat 保持作业定义；从官方 upstream zip
  恢复 20 条 hammer 示范。HW3 改用修复后的 v2 数据集并在报告中注明。
- **R**：两份作业的多种子实验与报告完整复现，差异写进 README。

## 9. 单卡云实验调度

- **S**：按小时计费的单卡云实例，需要同时跑 LLM RL（GPU）和上百个小 RL 实验（CPU）。
- **A**：GPU 用文件队列串行执行（`.sh → .running → done/`），保持一条主训练链路；
  CPU 多核并行跑 CQL/IQL/HER 的种子与绘图；所有指标落盘为 CSV/JSONL，绘图脚本
  只读原始数据生成图和 LaTeX 宏。
- **R**：GPU 主链路利用率接近 100%，CPU 实验与 GPU 实验并行完成；报告数字全部由
  脚本生成，没有手工录入。
- **追问**：多种子统计怎么聚合？（先对每个种子取最后若干次评估的均值，再跨种子求
  mean±std；不同种子步数不齐时按种子分别取尾部，否则会出现 NaN。）
