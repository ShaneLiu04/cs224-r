# 我的实现工作区

这个目录用于放**你自己的**作业实现与实验记录。默认是空的（只有带说明的占位文件），
请先独立完成后再提交任何代码。

## 使用方式

```powershell
# 从官方 starter code 复制一份到工作区（不要直接改 starter-code/）
Copy-Item -Recurse `
  assignments/spring2026/homework1-imitation/starter-code `
  solutions/spring2026/homework1-imitation/code
```

然后在 `code/` 中填写 TODO，把结果、曲线和 writeup 放在同一作业目录下：

```text
solutions/spring2026/homework1-imitation/
├── code/                    # 你的实现（从 starter-code 复制）
├── results/                 # 训练日志、评估 CSV、图表（已被 .gitignore 忽略部分大文件）
└── writeup.md               # 用 templates/assignment-writeup.md 的结构
```

## 学术诚信（务必阅读）

CS224R honor code 明确规定：

1. 不得复制、参考他人（包括 AI 生成）的解答与代码；
2. 不得查看往年官方解答或网上解答；
3. **不得把自己的作业解答公开发布**（如公开 git repo）。

因此：

- 如果你正在修读 CS224R：请保持本目录不公开（private repo 或不上传）；
- 已经修完/旁听学习：公开时请在 writeup 中注明这是学习实现，而非课程提交；
- 任何时候都不要把这里的代码当作课程提交。

## 建议的提交节奏

| 作业 | 先完成 | 再记录 |
| --- | --- | --- |
| HW1 | 四个 TODO + 四条命令 | 报告表格/曲线 + 失败分析 |
| HW2 | P1→P2→P3 | W&B CSV + UTD 消融结论 |
| HW3 | AWAC→IQL→stitching | 3-seed CSV + 对比表 |
| Project | SFT→IPO→RLOO | pass@k 图 + 扩展实验 + 报告 |

每完成一个 TODO，在 `PROGRESS.md` 里勾选并链接到你的 writeup。
