# 面试题库：补充专题

README 的「高频问题速查」覆盖课程主干。本页补充面试常问、但课程讲得少的专题，
每题给出可以 1 分钟讲完的答案与延伸。标 ★ 的题和本仓库实验直接相关，可以接
[PORTFOLIO](PORTFOLIO.md) 的故事。

## A. Off-policy 修正与重要性采样

**Q1. 重要性采样为什么方差大？** 权重是轨迹上逐步概率比的乘积 `Π ρ_t`，方差随长度
指数增长。常用手段：per-decision IS、截断权重、只在单步上做比值（PPO）。

**Q2. V-trace 做了什么？** IMPALA 中 actor 与 learner 策略不同步。V-trace 用截断比值
`c_t = min(c̄, ρ_t)` 构造多步目标，`ρ̄` 控制收敛到哪个策略的价值、`c̄` 控制方差；
`ρ̄ = c̄ = 1` 时在 on-policy 情形退化为 n-step return。Retrace 是 Q 函数版本。

**Q3. ★ PPO 为什么只用单步比值也能工作？** PPO 用状态分布近似不变的一阶近似，
只修正动作分布；clip 让比值离 1 太远的样本梯度为 0，等价于软信赖域。

**Q4. LLM RL 里训练引擎与 vLLM 的 log-prob 不一致怎么办？** 用训练引擎重算 old log-prob，
或对比值做截断重要性采样（TIS）；否则「on-policy」比值并不等于 1。

## B. SAC 与最大熵 RL

**Q5. SAC 的目标？** `max E[Σ r_t + α H(π(·|s_t))]`。软 Bellman 备份
`Q(s,a) = r + γ E[Q(s',a') − α log π(a'|s')]`，策略最小化 `E[α log π(a|s) − Q(s,a)]`。

**Q6. 为什么用两个 Q 取 min？** 抑制过估计（clipped double-Q，同 TD3）。2026 HW2 用
10 个 critic 中随机两个取 min（REDQ 风格），配合 UTD=5 提高样本效率。

**Q7. 自动温度？** 把 `α` 当作拉格朗日乘子，约束策略熵不低于目标值（常取 `−|A|`）。

**Q8. 重参数化与 tanh 修正？** `a = tanh(μ + σε)`，log-prob 要减去
`Σ log(1 − tanh²(u))` 的 Jacobian 项。

## C. RLHF 流水线

**Q9. 经典三阶段？** SFT → 奖励模型（Bradley-Terry：`−log σ(r(x,y_w) − r(x,y_l))`）→
PPO 最大化 `r(x,y) − β KL(π‖π_ref)`。PPO 阶段需要 actor、critic、reward、reference
四个模型。

**Q10. KL 惩罚加在哪里？** 常见做法是逐 token 加到奖励上（InstructGPT），或作为损失项
（GRPO）。KL 估计器：`k1 = log(π/π_ref)` 无偏但可为负；`k3 = r − 1 − log r`（`r = π_ref/π`）
无偏且非负，方差更小，GRPO 用 k3。

**Q11. ★ DPO 的隐式奖励和局限？** 奖励为 `β log(π/π_ref)`，免去奖励模型与采样。局限：
离线数据、对数据分布外回答无约束、可能同时压低 chosen 与 rejected 的概率。IPO 用平方
损失避免偏好接近确定时的过拟合；本仓库发现序列求和的 IPO 不稳定，长度平均更稳。

**Q12. Reward hacking 怎么监控？** 拆分奖励构成、监控长度与 KL、留出人工/规则评测、
早停；奖励模型集成或不确定性惩罚。

## D. LLM 推理 RL

**Q13. ★ GRPO、RLOO、Dr. GRPO 的区别？**

| 方法 | 优势 | 归一化 |
| --- | --- | --- |
| RLOO | `r_i − mean_{j≠i} r_j` | 无标准差归一化，无偏 |
| GRPO | `(r_i − mean_j r_j) / std_j r_j` | 除以组内标准差；损失按序列长度平均 |
| Dr. GRPO | `r_i − mean_j r_j` | 去掉 std 与长度归一化，避免偏向难度极端的题和长错误回答 |

除以 std 会放大「几乎全对或全错」组的权重；按长度平均会让错误回答越长、每 token
惩罚越小，导致回答变长。

**Q14. ★ RL 到底让模型学到了新能力吗？** Yue et al. 2025 发现 RLVR 提升 pass@1，但大 k 时
pass@k 可能不如基座，即主要是 sharpening。本仓库长程 RLOO 的 pass@1 从 0.24 到 0.505，
pass@4 只从 0.445 到 0.60，与此一致。

**Q15. 全对/全错的组怎么办？** 优势全为 0，梯度浪费。DAPO 做动态采样过滤这些组；
课程学习按通过率选题。

**Q16. 测试时计算怎么用？** 多数投票（self-consistency）、验证器重排（best-of-n）、
搜索。比较方法时画「准确率 vs 采样数」曲线，而不是只报单点。

## E. 模型基 RL

**Q17. MBPO 的核心？** 学集成动力学模型，从真实状态出发做短 rollout（k 步）扩充数据，
再用 SAC 训练。短 rollout 控制复合误差，集成提供不确定性。

**Q18. Dreamer 与 MuZero？** Dreamer 在学到的潜空间里想象轨迹并训练 actor-critic；
MuZero 只学对规划有用的价值、策略与奖励，不重建观测，配合 MCTS。

**Q19. 模型被 exploit 怎么办？** 不确定性惩罚（MOPO）、限制 rollout 长度、集成分歧。

## F. 离线到在线与探索

**Q20. ★ 离线 RL 的数据从哪里来最好？** 不一定是「探索得最好」的数据。本仓库 2025 HW3：
IQL 在 RND 数据上更好，CQL 在随机数据上更好；RND 数据覆盖广但每状态动作稀疏，
强保守正则会把策略钉在差动作上。

**Q21. 离线到在线微调的难点？** 保守 Q 值在线初期被修正，策略先掉后升（unlearning）。
Cal-QL 校准保守值；或平衡离线与在线回放比例。

## G. 系统与工程

**Q22. FSDP 与 DDP？** DDP 每卡完整副本、梯度 all-reduce；FSDP（≈ ZeRO-3）分片参数、
梯度与优化器，前向/反向时按层 all-gather，以通信换显存。

**Q23. 张量并行与流水线并行？** TP 切分矩阵乘法，层内通信频繁，适合机内 NVLink；PP 按层
切分，有气泡，用 micro-batch 填充。

**Q24. vLLM 为什么快？** PagedAttention 按块管理 KV cache 消除碎片，连续批处理让请求随时
进出，CUDA graph 降低启动开销，前缀缓存复用共享提示。

**Q25. ★ 同卡训练与推理冲突？** caching allocator 不归还显存。解决：串行、同进程
sleep/wake、offload 训练状态，或分离部署。

**Q26. bf16 与 fp16？** 见 [ENGINEERING §5](ENGINEERING.md#5-混合精度)。

## H. 研究与实验设计

**Q27. 怎么判断一个改进是真的？** 多种子 mean±std、相同计算预算、单变量消融、
报告失败的设置；对比基线要用作者的最佳超参。

**Q28. ★ 读 starter code 时发现异常怎么办？** 先用最小实验证实（固定种子比较开关前后
的数据），保留原行为作为默认，新增开关，报告中写清楚。
