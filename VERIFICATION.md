# 仓库完整性检查

最后检查：2026-10-03。

## 材料清单

| 目录 | 期望 | 实际 |
| --- | --- | --- |
| `slides/spring2026` | 17 份 PDF（01–13、15–18） | 17 |
| `slides/spring2025` | 19 份 PDF（01–18 + Q-learning section） | 19 |
| `slides/spring2023` | 19 份 PDF | 19 |
| `assignments/spring2026/homework1-imitation` | PDF + tex + starter-code（6 个 TODO/入口文件 + assets） | 已核对 |
| `assignments/spring2026/homework2-online-rl` | PDF + tex + starter-code（gridworld/on-policy/off-policy/modal/cfgs/tests） | 已核对 |
| `assignments/spring2026/homework3-offline-rl` | PDF + tex + starter-code（cs224r 包 + offline_datasets） | 已核对 |
| `assignments/spring2026/default-project` | 项目指南 + starter-code（sft/ipo/rloo/evaluation） | 已核对 |
| `assignments/spring2025` | 4 次作业（HW3 含 updated 版与 v1 版） | 已核对 |
| `assignments/spring2023` | 4 次作业 + 项目指南 | 已核对 |
| `materials/spring2026` | compute guide、exam review、TA section、project index | 6 个文件 |

## 数据与夹具保留

以下二进制不会进入 `.gitignore` 的忽略范围，因为官方 starter code 依赖它们：

- `assignments/spring2023/homework1-imitation/starter-code/cs224r/expert_data/*.pkl`
  （及 2025 版的同名文件）；
- `assignments/spring2023/homework2-online-rl/starter-code/ac/demos/*.npz`
  （及 2025 版同名文件）；
- `assignments/spring2026/homework3-offline-rl/starter-code/offline_datasets/*.npz`。

## 已知缺失

- 官方项目报告 PDF（约 570 篇）未入库，只保留索引页；
- 官方视频（2025 版）只在 YouTube，未下载；
- 作业答案（官方不公开，本仓库也不包含）。

## 已知上游小问题（不在本仓库修复范围内）

- 2023/2025 HW1 官方 starter README 中的 `cs224r/scripts/run_hw1.ipynb` 链接在
  zip 内不存在（上游打包遗漏）；
- 2026 材料文件名 `CS224R_Tutotial_Max.pdf` 含官方拼写错误，为保真未改名。

## 复验命令

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check-repo.ps1
```
