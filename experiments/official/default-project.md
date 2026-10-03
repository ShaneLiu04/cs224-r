# 官方题目索引：Default Project（Countdown RL）

- 指南：[CS224R_Default_Project_Guidelines.pdf](../../assignments/spring2026/default-project/CS224R_Default_Project_Guidelines.pdf)、
  [Custom 指南](../../assignments/spring2026/default-project/CS224R_Custom_Project_Guidelines.pdf)
- 代码：[starter-code/](../../assignments/spring2026/default-project/starter-code/)
- 任务说明：starter README（非常详细，强烈建议先读）
- 项目示例索引：[materials/spring2026/projects-index.html](../../materials/spring2026/projects-index.html)

## 管线拆解

| 阶段 | 目标 | TODO 位置 | 关键点 |
| --- | --- | --- | --- |
| SFT | 从正确样例学格式与推理 | `sft_trainer/sft.py::train` | response-only loss |
| IPO | 从偏好对学相对好坏 | `ipo_trainer/ipo.py::train` | `Δ_w−Δ_l` 与参考模型 |
| RLOO | 从可验证奖励提升 pass@k | `rloo_trainer/rloo_update_worker.py::update` | LOO 基线、熵/KL |
| 评估 | 量化 pass@k 与格式正确率 | `evaluation/view_passk.ipynb` | 多温度/多 k |

## Countdown 任务

- 输入：目标数 + 可用数字；
- 输出：`<answer>等式</answer>`；
- 奖励：0.0 无答案 / 0.1 格式对但结果错 / 1.0 完全正确；
- 难度可由数字个数与目标范围控制（适合做课程学习）。

## 扩展方向（指南 Section 2）

- 难度课程：从简单题到难题的调度；
- 部分解答/已作答问题的探索；
- 工具使用（生成可执行验证）；
- 过程奖励与稠密信号；
- 测试时验证与重采样。

## 隐藏的思考题

1. SFT、IPO、RLOO 三者的数据需求与目标函数差异？
2. RLOO 的优势在组内奖励全同（全对/全错）时是多少？如何设计课程避免？
3. pass@k 与 pass@1 分别衡量什么？为什么两者都要报告？
4. 与 RLHF 的 PPO 相比，RLOO 省掉了什么、付出了什么？
