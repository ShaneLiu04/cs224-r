# Official upstream snapshot

本仓库中的全部原始课程材料来自官方课程网站仓库：

| 项 | 值 |
| --- | --- |
| 上游仓库 | https://github.com/cs224r/cs224r.github.io |
| 固定 commit | `3f8136a624dbb51ce10dbb78a7b5bf2575d2671a` |
| commit 日期 | 2026-09-15 |
| 同步日期 | 2026-10-03 |
| 官方课程主页 | https://cs224r.stanford.edu/ |

使用 git 稀疏检出（仅下载所需 blob）后，把官方文件复制为以下本地结构：

```text
slides/spring2026/      <- slides/            （17 份，2026 版无第 14 讲 exploration）
slides/spring2025/      <- spring_2025/slides （19 份，含 Q-learning section）
slides/spring2023/      <- spring_2023/slides （19 份）
assignments/spring2026/ <- material/          （HW1-3 + default project）
assignments/spring2025/ <- spring_2025/material/
assignments/spring2023/ <- spring_2023/material/
materials/spring2026/   <- material/          （compute guide、exam review、TA section、project index）
materials/spring2025/   <- spring_2025/projects/（project index）
```

## 处理说明

1. **starter code zip 已解压**：官方 zip 内的 `__MACOSX/`、`.DS_Store`、`__pycache__/`、`*.pyc` 已清除，内容按作业目录整理为 `starter-code/`。原始 zip 可从上游固定 commit 下载。
2. **未导入项目报告 PDF**：上游 `projects/pdfs/`（2026 届 244 篇）与 `spring_2025/projects/pdfs/`（326 篇）合计约 1GB，属于学生作品而非课程讲义。本仓库只保留项目索引页 `materials/*/projects-index.html`，PDF 通过课程主页链接访问。
3. **未导入网站资源**：`css/`、`js/`、`images/`、`CNAME` 等站点文件未复制。
4. **未导入**：`CS224R_Tutotial_Max.key` 已导入（14.7MB，与同名 PDF 内容重复，供 macOS 学员使用）。

## 版本差异（重要）

- **Spring 2026（当前版）**：3 次作业（10% + 15% + 15%）、期中考试 25%、项目 35%；无第 14 讲 exploration；无 HW4。
- **Spring 2025**：4 次作业、项目 50%；有第 14 讲 exploration、第 16 讲 autonomous learning；B 站/YouTube 有完整录像。
- **Spring 2023**：4 次作业、项目 50%；作业与 2025 基本一致（2025 是 2023 的修订版），多出 practical deep RL、transfer learning、skill discovery 等讲义。

作业接口三代之间不通用，做作业请以对应版本的 handout 与 starter code 为准。

## 版权与许可

官方材料版权归 Stanford University 与课程团队（© Chelsea Finn）所有，本仓库仅作个人学习镜像，不做商业使用。
本仓库原创笔记、索引与脚本采用 MIT License。仓库不包含任何作业答案；再次强调 honor code 禁止公开发布作业解答。
