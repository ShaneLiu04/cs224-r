"""Held-out Countdown prompts from the tail of the train split.

The SFT builder samples train rows with stride len/20000 (indices < 480k) and
RL reads from the head of a shuffled prefix, so the last rows are unseen by
every trainer here. The official test split has only 50 prompts; this set
gives tighter confidence intervals.
"""
import argparse
import os
from collections import Counter

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/root/autodl-tmp/data/countdown/train.parquet")
    parser.add_argument("--dest", default="/root/autodl-tmp/data/countdown_heldout")
    parser.add_argument("--size", type=int, default=500)
    args = parser.parse_args()
    table = pq.read_table(args.source).replace_schema_metadata(None)
    tail = table.slice(table.num_rows - args.size)
    os.makedirs(args.dest, exist_ok=True)
    pq.write_table(tail, os.path.join(args.dest, "test.parquet"))
    head = table.slice(0, 20000).column("nums").to_pylist()
    print("heldout", tail.num_rows,
          "nums-count in heldout", Counter(len(x) for x in tail.column("nums").to_pylist()),
          "in first 20k train", Counter(len(x) for x in head))


if __name__ == "__main__":
    main()
