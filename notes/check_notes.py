"""Check that every lecture note exists, has the required sections, and links resolve."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REQUIRED_SECTIONS = [
    "## 学习目标",
    "## 核心概念",
    "## 实现映射",
    "## 易错点与反例",
    "## 高频问题",
    "## 本讲小结",
]

LECTURES = [
    "lecture-01-intro-mdps.md",
    "lecture-02-imitation-learning.md",
    "lecture-03-policy-gradients.md",
    "lecture-04-actor-critic.md",
    "lecture-05-off-policy-actor-critic.md",
    "lecture-06-q-learning.md",
    "lecture-07-offline-rl.md",
    "lecture-08-reward-learning.md",
    "lecture-09-rlhf-preference-optimization.md",
    "lecture-10-rl-llm-reasoning.md",
    "lecture-11-model-based-rl.md",
    "lecture-12-multi-task-goal-conditioned-rl.md",
    "lecture-13-meta-rl.md",
    "lecture-14-exploration.md",
    "lecture-15-hierarchical-rl-il.md",
    "lecture-16-sim-to-real.md",
    "lecture-17-rl-for-vlas.md",
    "lecture-18-frontiers.md",
]

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]*)?\)")


def main() -> int:
    root = Path(__file__).resolve().parent
    failures: list[str] = []

    for name in LECTURES:
        path = root / name
        if not path.exists():
            failures.append(f"missing note: {name}")
            continue
        text = path.read_text(encoding="utf-8")
        for section in REQUIRED_SECTIONS:
            if section not in text:
                failures.append(f"{name}: missing section {section}")
        for target in LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (root / target).resolve()
            if not resolved.exists():
                failures.append(f"{name}: broken link {target}")

    if failures:
        print("\n".join(failures))
        print(f"\nFAILED: {len(failures)} issue(s)")
        return 1

    print(f"OK: {len(LECTURES)} notes, sections and links verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
