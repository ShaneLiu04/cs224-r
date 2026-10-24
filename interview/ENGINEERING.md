# RL / LLM-RL 工程与调试手册

面向 ML 工程师面试的「怎么把实验跑对、跑快、跑得可信」。每一节先给可直接复述的结论，
再给本仓库里的真实例子。算法推导见 [notes/](../notes/README.md)，故事版见
[PORTFOLIO](PORTFOLIO.md)。

## 1. 调试顺序：先证明实现对，再调超参

1. **解析检查**：找一个不依赖训练的恒等式。策略等于参考策略时 IPO 损失为 `(1/(2β))²`、
   DPO 为 `log 2`；均匀分布的熵为 `log|V|`、KL 为 0；GAE 在 `λ=0` 时退化为 TD 残差。
2. **手算小例子**：3 步轨迹手算 GAE 与 returns，写成单元测试。
3. **过拟合单个 batch**：SFT 能把一个 batch 的损失压到接近 0，否则是掩码或数据问题。
4. **已知最优的玩具环境**：Gridworld 最优路径、bit-flip 6 bit、meta-RL 网格最优 0.7。
5. **读环境和 starter**：2025 HW3 的数据采集器其实是 oracle（见 PORTFOLIO 故事 4）。
6. **最后才调超参**，每次只动一个变量，至少 3 个种子。

## 2. LLM RL 常见 bug 清单

| 症状 | 常见原因 | 检查方法 |
| --- | --- | --- |
| 样本越过答案标签继续生成 | 停止 token 只登记了一种 BPE 切分 | 打印模型实际生成的 token id |
| SFT 损失低但输出乱 | 提示被重复套用 chat 模板，或 label 没右移 | decode 一条训练样本的 input 与 label |
| 梯度全为 0 | `max_grad_norm=0` 被当成「裁剪到 0」 | 区分「不裁剪」与「裁剪范数」 |
| 短跑学习率一直很小 | 调度器按全数据估计步数，停在 warmup | 设置 `max_steps` 时按上限算调度 |
| 偏好损失下降但准确率 0.5 | 序列求和的 `Δ` 被长回答主导 | 改长度平均，同时监控 accuracy |
| KL 突然爆炸 | 参考模型与策略的 tokenizer/模板不一致，或 bf16 下 log-prob 精度不足 | 用 fp32 算 log-softmax，比较两者在同一序列上的 log-prob |
| 奖励涨、真实正确率不涨 | reward hacking（格式分、长度） | 拆开奖励构成画图 |
| 一组里全对或全错仍在更新 | 优势按整个 batch 归一化，而不是按 prompt | GRPO / RLOO 的均值和标准差按组算 |
| KL 项让策略加速偏离参考 | KL 的符号反了，惩罚变成了奖励 | 参考策略处 KL 应为 0，偏离后损失应上升 |
| pad 被学进模型 | label 没把 padding 位置标成忽略 | 检查 loss mask 在 pad 上是否为 0 |

## 3. 显存账

按 `N` 个参数估算，面试时把乘法写出来。下面用 Qwen2.5-0.5B 的数量级：`N ≈ 5×10^8`，
24 层，14 个注意力头，2 个 KV 头，头维度 64，词表 151936。

**一个带 AdamW 的 bf16 模型，静态部分：**

| 张量 | 字节 | 0.5B |
| --- | --- | --- |
| bf16 权重 | `2N` | 1 GB |
| fp32 主权重 | `4N` | 2 GB |
| 一阶矩、二阶矩 | `4N + 4N` | 4 GB |
| bf16 梯度 | `2N` | 1 GB |
| 合计 | `16N` | 8 GB |

参考模型不更新，只有 1 GB 权重。7B 按同样的 `16N` 约 112 GB，单卡放不下，要 ZeRO 或 FSDP。

**容易被错当成权重的两块：**

- logits：`B × T × |V| × 4` 字节（float32）。`B=4`、`T=512` 约 1.2 GB；`T=4096` 约 10 GB。
  只对需要的 token 做分块 log-sum-exp，不要留整张表。
- 朴素注意力分数：一层是 `B × H × T × T × 2` 字节。`B=4`、`T=4096`、24 层都保存给反传时大约几十 GB。
  scaled dot-product attention 不算完整张分数矩阵。`T=512` 时这一项不大，所以短序列 OOM 先查 logits。
- KV cache：`2 × L × n_kv × d × T × B × 2` 字节。上面的配置、`T=2048`、一条序列大约 24 MB。
  本仓库训练进程留下的约 89 GB 是激活、logits 和 caching allocator，不是 KV cache。

vLLM 启动时按 `gpu_memory_utilization × 整卡容量` 预留，不读取别的进程的占用。
`torch.cuda.empty_cache()` 只释放调用者自己的缓存。同卡要交替训练和推理时，放在同一进程里
offload 训练状态并 sleep 推理引擎（veRL、OpenRLHF 的做法），或者干脆串行。

## 4. 吞吐

- LLM RL 的瓶颈通常是**采样**而不是反向传播：生成是逐 token 的访存受限计算。
  手段：vLLM/SGLang 连续批处理、提高组大小、限制最大生成长度、异步 rollout。
- 小网络 RL（MuJoCo/网格）瓶颈在**环境步进**（CPU）：多进程向量化环境，
  多种子并行占满 CPU 核，GPU 留给大模型。
- 衡量：tokens/s、env steps/s、GPU SM 利用率与显存占用一起看；利用率高但
  tokens/s 低往往是 padding 浪费。

## 5. 混合精度

- **bf16** 指数位与 fp32 相同，不需要 loss scaling，是训练首选；尾数只有 7 位，
  累加、softmax、log-prob、KL 应在 fp32 中做。
- **fp16** 动态范围小，需要动态 loss scaling；出现 `inf/nan` 时先查 scaler 是否频繁回退。
- 重要性比值 `exp(logp_new − logp_old)` 对精度敏感：训练引擎与推理引擎（vLLM）的
  log-prob 存在数值差异，PPO/GRPO 中 on-policy 比值不再严格为 1，需要截断或用训练
  引擎重算 old log-prob。

## 6. 评测可信度

- **pass@k 用无偏估计**：`1 − C(n−c, k)/C(n, k)`，`n` 为每题样本数，`c` 为正确数；
  不要用「采 k 次看有没有对」的单次估计。
- **多种子**：报告 mean ± std 与种子数；单种子的离群值（如 IQL ζ=0.9 某种子 −141.6）
  会把均值拉偏，图中保留每个种子。
- **按种子先聚合**：不同种子评估步数不齐时，先对每个种子取尾部均值，再跨种子统计。
- **测试集隔离**：Countdown 测试题不进训练；留出集用于选检查点，测试集只报一次。
- **同口径比较**：方法 A 的 pass@1 不和 方法 B 的 pass@8 放在一个格子里比。
- **数字只来自脚本**：原始 CSV/JSONL → 绘图脚本 → 图与 LaTeX 宏 → 报告。
- **种子步数不齐**：先在每个种子内部取尾部均值，再跨种子求 mean±std。全局 `dropna` 后取最后几行，
  会只剩下跑得最久的种子。

## 7. 实验管理

- 每个运行一个目录：配置、种子、git commit、原始指标、检查点。
- GPU 文件队列：脚本按名称排序执行，`.sh → .running → done/`，日志单独存放；
  需要插队时改文件名前缀。
- 云实例计费：任务完成即归档与关机；GPU 空闲时立即排下一批或关机。
- 跨平台：PowerShell 会吞掉内层双引号并展开 `$VAR`，远程命令一律写成脚本文件上传后
  执行，并去掉 Windows 换行（`sed -i 's/\r$//'`）。

## 8. 分布式 LLM RL 架构（概念）

| 组件 | 作用 | 代表实现 |
| --- | --- | --- |
| Rollout 引擎 | 高吞吐采样，PagedAttention、连续批处理 | vLLM、SGLang |
| 训练引擎 | 参数/梯度/优化器分片 | FSDP、DeepSpeed ZeRO-1/2/3、Megatron TP/PP |
| 编排 | 多角色（actor/ref/reward/critic）调度 | Ray（OpenRLHF、veRL） |
| 权重同步 | 训练后把新权重推给推理引擎 | NCCL broadcast、共享显存 colocate |

- **Colocate vs 分离部署**：同卡交替训练与推理省卡但要切换显存；分离部署可流水线化，
  但存在策略滞后（off-policy），需要重要性采样修正。
- **ZeRO 阶段**：1 分片优化器状态，2 再分片梯度，3 再分片参数（每层前 all-gather）。
