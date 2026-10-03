# 主题：LLM 的 RL（RLHF / DPO / 推理 RL）

> 对应讲次：[L9](../../notes/lecture-09-rlhf-preference-optimization.md)、
> [L10](../../notes/lecture-10-rl-llm-reasoning.md)；
> 作业：[Default Project](../../assignments/spring2026/default-project/README.md)。

## 问题脉络

1. 语言模型 = 策略；token = 动作；奖励 = RM/规则；
2. RLHF：SFT → RM → PPO + KL；
3. 偏好替代方案：DPO、IPO、KTO；
4. 可验证奖励 + 组内相对优势：GRPO、RLOO；
5. 测试时算力：CoT、采样、验证、搜索；
6. 评估转变：score vs compute、pass@k。

## 必读论文

| 论文 | 年份 | 为什么读 |
| --- | --- | --- |
| Christiano et al. | 2017 | 偏好学习基础 |
| Ouyang et al., *InstructGPT* | 2022 | RLHF 全流程 |
| Rafailov et al., *DPO* | 2023 | 隐式奖励 |
| Azar et al., *IPO* | 2024 | 防过拟合偏好 |
| Ethayarajh et al., *KTO* | 2024 | 前景理论视角 |
| Shao et al., *DeepSeekMath (GRPO)* | 2024 | 组内优势 |
| DeepSeek-AI, *R1* | 2025 | 大规模推理 RL |
| Ahmadian et al., *RLOO* | 2024 | leave-one-out 基线 |
| Wei et al., *CoT* / Wang et al., *Self-Consistency* | 2022 | 推理增强 |
| Brown, *test-time compute*（slides） | 2024–26 | 新 scaling 维度 |

## 公式备忘

```text
RLHF:  max_π E[R(x,y)] − β KL(π || π_ref)
最优:   π*(y|x) ∝ π_ref(y|x) exp(R/β)
DPO:   L = −log σ(β[(logπ_θ(y_w)−logπ_ref(y_w)) − (logπ_θ(y_l)−logπ_ref(y_l))])
GRPO:  Â_i = (r_i − mean(r)) / std(r)
RLOO:  Â_i = r_i − mean_{j≠i}(r_j)
pass@k = 1 − C(n−c,k)/C(n,k)
```

## 常见实验（项目直接可用）

- SFT 数据量/质量对 RLOO 冷启动的影响；
- `group_size`、温度、熵/KL 系数对稳定性；
- 课程学习（难度调度）与 pass@k；
- 长度惩罚与 token 预算；
- IPO vs DPO；RLOO vs GRPO；
- 测试时验证/重采样策略。

## 开放问题

- 无 critic 方法（GRPO/RLOO）的方差与偏差理论；
- 奖励模型过优化与安全；
- 长时域 agentic RL 的信用分配；
- 多轮工具使用的 RL（RL with tools）。

## 本地材料

- slides 09–10；默认项目 starter code（SFT/IPO/RLOO/eval）；
- [catalog/default-project.md](../catalog/default-project.md)。
