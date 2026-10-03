"""Validate SOURCES.yaml and print a summary.

Usage: python experiments/scripts/check_sources.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED = ["id", "title", "url", "author", "accessed", "version", "topics", "type", "license", "confidence"]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    path = root / "SOURCES.yaml"
    raw = path.read_text(encoding="utf-8").strip()
    # SOURCES.yaml is a JSON array stored with a .yaml name for tooling compatibility.
    data = json.loads(raw)

    failures: list[str] = []
    ids: set[str] = set()
    urls: list[str] = []
    for i, entry in enumerate(data):
        for field in REQUIRED:
            if field not in entry:
                failures.append(f"entry #{i}: missing field {field}")
        if entry.get("id") in ids:
            failures.append(f"duplicate id: {entry.get('id')}")
        ids.add(entry.get("id", ""))
        url = entry.get("url", "")
        if not url.startswith(("http://", "https://")):
            failures.append(f"{entry.get('id')}: invalid url {url!r}")
        urls.append(url)

    dupes = [u for u, c in Counter(urls).items() if c > 1]
    if dupes:
        failures.append(f"duplicate urls: {dupes}")

    print(f"entries: {len(data)}")
    print("types:", dict(Counter(e.get("type") for e in data)))
    print("confidence:", dict(Counter(e.get("confidence") for e in data)))
    print("mirrored:", sum(1 for e in data if e.get("mirror")))

    if failures:
        print("\n".join(failures))
        print(f"FAILED: {len(failures)} issue(s)")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
