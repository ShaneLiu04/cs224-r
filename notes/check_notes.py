"""Strict validator for the CS224R lecture notes.

Checks performed
----------------
1. All 18 lecture files exist and cover the 2026/2025 lecture map.
2. Every note has the required metadata and section structure.
3. Core algorithm lectures additionally have a `## 深入推导` section.
4. Every note contains at least two fenced formula blocks.
5. Every note has an interview follow-up chain, self-test, and >=2 cross links.
6. Relative links (slides, other notes, repo files) resolve on disk.
7. notes/README.md indexes every lecture; GLOSSARY/ROADMAP exist and cover
   the required key terms.
8. No TODO/TBD placeholders remain.

Usage
-----
    python notes/check_notes.py            # concise summary
    python notes/check_notes.py --verbose  # per-note table
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- expectations -----------------------------------------------------------

LECTURES = {
    "lecture-01-intro-mdps.md": "L01",
    "lecture-02-imitation-learning.md": "L02",
    "lecture-03-policy-gradients.md": "L03",
    "lecture-04-actor-critic.md": "L04",
    "lecture-05-off-policy-actor-critic.md": "L05",
    "lecture-06-q-learning.md": "L06",
    "lecture-07-offline-rl.md": "L07",
    "lecture-08-reward-learning.md": "L08",
    "lecture-09-rlhf-preference-optimization.md": "L09",
    "lecture-10-rl-llm-reasoning.md": "L10",
    "lecture-11-model-based-rl.md": "L11",
    "lecture-12-multi-task-goal-conditioned-rl.md": "L12",
    "lecture-13-meta-rl.md": "L13",
    "lecture-14-exploration.md": "L14",
    "lecture-15-hierarchical-rl-il.md": "L15",
    "lecture-16-sim-to-real.md": "L16",
    "lecture-17-rl-for-vlas.md": "L17",
    "lecture-18-frontiers.md": "L18",
}

# Lectures that must include a full step-by-step derivation appendix.
# L01 (course/MDP overview) and L18 (research methodology) are exempt.
DEEP_LEARNINGS = {
    "L02", "L03", "L04", "L05", "L06", "L07", "L08", "L09", "L10",
    "L11", "L12", "L13", "L14", "L15", "L16", "L17",
}

REQUIRED_SECTIONS = [
    "## 摘要",
    "## 一页速览卡",
    "## 学习目标",
    "## 先修知识",
    "## 核心概念",
    "## 关键推导",
    "## 实现映射",
    "## 易错点与反例",
    "## 实验或练习",
    "## 高频问题",
    "## 面试追问链",
    "## 自测题",
    "## 关联笔记",
    "## 本讲小结",
    "## 延伸阅读",
]

REQUIRED_METADATA = ["- 课程版本", "- 日期", "- 官方材料", "- 状态"]

MIN_LINES = 200
MIN_FORMULA_BLOCKS = 2
MIN_RELATED_LINKS = 2

KEY_GLOSSARY_TERMS = [
    "MDP",
    "策略梯度",
    "GAE",
    "PPO",
    "SAC",
    "Q-learning",
    "IQL",
    "expectile",
    "DPO",
    "GRPO",
    "RLOO",
    "HER",
    "PEARL",
    "RND",
    "sim2real",
    "VLA",
]

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]*)?\)")
FORMULA_RE = re.compile(r"```(?:text|python|bash)\b", re.IGNORECASE)
# Only flag bare placeholder markers; notes legitimately discuss assignment TODOs.
FORBIDDEN = re.compile(r"^\s*[-*#>]*\s*(TODO|TBD)\s*[:：]?\s*$|待补|待写", re.MULTILINE)


def check_note(path: Path) -> tuple[dict[str, int], list[str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    issues: list[str] = []

    for section in REQUIRED_SECTIONS:
        if section not in text:
            issues.append(f"缺少章节 {section}")

    for meta in REQUIRED_METADATA:
        if meta not in text:
            issues.append(f"缺少元信息 {meta}")

    if len(lines) < MIN_LINES:
        issues.append(f"行数 {len(lines)} < {MIN_LINES}")

    formula_blocks = len(FORMULA_RE.findall(text))
    if formula_blocks < MIN_FORMULA_BLOCKS:
        issues.append(f"公式块 {formula_blocks} < {MIN_FORMULA_BLOCKS}")

    if "## 关联笔记" in text:
        block = text.split("## 关联笔记", 1)[1].split("\n## ", 1)[0]
        links = [t for t in LINK_RE.findall(block) if t.endswith(".md")]
        if len(links) < MIN_RELATED_LINKS:
            issues.append(f"关联笔记链接 {len(links)} < {MIN_RELATED_LINKS}")

    hit = FORBIDDEN.search(text)
    if hit:
        issues.append(f"存在占位符 {hit.group(0)!r}")

    stats = {
        "lines": len(lines),
        "formulas": formula_blocks,
        "questions": text.count("?") + text.count("？"),
    }
    return stats, issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print per-note stats")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    repo = root.parent
    issues: list[str] = []

    if args.verbose:
        print(f"{'lecture':<48}{'lines':>6}{'formulas':>9}{'questions':>10}")
        print("-" * 74)

    for name, label in LECTURES.items():
        path = root / name
        if not path.exists():
            issues.append(f"{label}: 文件缺失 {name}")
            continue
        stats, note_issues = check_note(path)
        if label in DEEP_LEARNINGS and "## 深入推导" not in path.read_text(encoding="utf-8"):
            note_issues.append("核心算法讲缺少 ## 深入推导")
        if args.verbose:
            print(f"{name:<48}{stats['lines']:>6}{stats['formulas']:>9}{stats['questions']:>10}")
        for item in note_issues:
            issues.append(f"{label}: {item}")

        # slides / repo links resolve
        for target in LINK_RE.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                issues.append(f"{label}: 失效链接 {target}")

    # README indexes every lecture
    readme = root / "README.md"
    if not readme.exists():
        issues.append("notes/README.md 缺失")
    else:
        readme_text = readme.read_text(encoding="utf-8")
        for name in LECTURES:
            if name not in readme_text:
                issues.append(f"notes/README.md 未索引 {name}")

    # glossary / roadmap
    glossary = root / "GLOSSARY.md"
    if not glossary.exists():
        issues.append("GLOSSARY.md 缺失")
    else:
        glossary_text = glossary.read_text(encoding="utf-8")
        missing = [t for t in KEY_GLOSSARY_TERMS if t.lower() not in glossary_text.lower()]
        if missing:
            issues.append(f"GLOSSARY 缺少关键术语: {', '.join(missing)}")

    if not (root / "READING-ROADMAP.md").exists():
        issues.append("READING-ROADMAP.md 缺失")

    if issues:
        print("\n".join(f"  - {item}" for item in issues))
        print(f"\nFAILED: {len(issues)} issue(s)")
        return 1

    print(f"OK: {len(LECTURES)} notes, structure / formulas / links / index / glossary verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
