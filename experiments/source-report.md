# 来源检查报告

生成方式：`python experiments/scripts/check_sources.py`（统计 `SOURCES.yaml`）。

## 最近一次检查

- 日期：2026-10-03
- 条目总数：61
- 按类型：official 10 / official-adjacent 1 / paper 40 / book 1 / course 1 /
  essay 1 / community 7
- 按可信度：primary 53 / secondary 6 / spoiler 2
- 缺少 license 字段：0
- 重复 URL：0

## 说明

- `mirror: true` 的官方条目已在 `slides/`、`assignments/`、`materials/` 中落地，
  以 `UPSTREAM.md` 的 commit 为准；
- `confidence: spoiler` 的条目仅作索引，修课时不要打开；
- 完整性由 `scripts/check-repo.ps1` 与本脚本共同校验。
