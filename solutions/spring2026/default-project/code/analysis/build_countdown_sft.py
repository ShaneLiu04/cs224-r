"""Build Countdown SFT pairs by searching exact arithmetic expressions.

The training prompts already contain the chat prefix. The completion is the
continuation that the response-only loss should raise: a short derivation
and an <answer> span. Search is exhaustive over permutations, operators,
and a fixed set of bracketings, so a returned expression uses each given
number once.
"""
import itertools
import os
import pyarrow as pa
import pyarrow.parquet as pq


OPS = ["+", "-", "*", "/"]
PATTERNS = {
    2: ["({a} {op1} {b})"],
    3: ["(({a} {op1} {b}) {op2} {c})", "({a} {op1} ({b} {op2} {c}))"],
    4: [
        "((({a} {op1} {b}) {op2} {c}) {op3} {d})",
        "(({a} {op1} ({b} {op2} {c})) {op3} {d})",
        "(({a} {op1} {b}) {op2} ({c} {op3} {d}))",
        "({a} {op1} (({b} {op2} {c}) {op3} {d}))",
        "({a} {op1} ({b} {op2} ({c} {op3} {d})))",
    ],
}


def _value(expr):
    if not all(ch in "0123456789+-*/(). " for ch in expr):
        return None
    try:
        return eval(expr, {"__builtins__": None}, {})
    except Exception:
        return None


def solve(numbers, target):
    """Return one exact expression, or None when the search finds nothing."""
    nums = [int(n) for n in numbers]
    letters = ["a", "b", "c", "d"][:len(nums)]
    patterns = PATTERNS.get(len(nums), [])
    for perm in itertools.permutations(nums):
        binding = dict(zip(letters, [str(n) for n in perm]))
        for ops in itertools.product(OPS, repeat=len(nums) - 1):
            fields = dict(binding)
            for index, op in enumerate(ops, start=1):
                fields[f"op{index}"] = op
            for pattern in patterns:
                expr = pattern.format(**fields)
                value = _value(expr)
                if value is not None and abs(value - target) < 1e-6:
                    return expr
    return None


def completion_for(expr):
    """Wrap a gold expression in the tags the reward function reads."""
    return f"<think> {expr} uses each given number once. </think>\n<answer> {expr} </answer>"


def _solve_row(row):
    prompt, target, numbers = row
    expr = solve(numbers, int(target))
    if expr is None:
        return None
    return prompt, completion_for(expr), int(target)


def main():
    import argparse
    from multiprocessing import Pool
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/root/autodl-tmp/data/countdown/train.parquet")
    parser.add_argument("--dest", default="/root/autodl-tmp/data/countdown_sft")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    table = pq.read_table(args.source, columns=["prompt", "target", "nums"])
    total = table.num_rows
    step = max(total // args.limit, 1)
    rows = []
    for index in range(0, total, step):
        if len(rows) >= args.limit:
            break
        rows.append((
            table.column("prompt")[index].as_py(),
            int(table.column("target")[index].as_py()),
            table.column("nums")[index].as_py(),
        ))
    prompts, completions, targets = [], [], []
    missed = 0
    with Pool(args.workers) as pool:
        for item in pool.imap_unordered(_solve_row, rows, chunksize=16):
            if item is None:
                missed += 1
                continue
            prompts.append(item[0])
            completions.append(item[1])
            targets.append(item[2])
            if len(prompts) % 2000 == 0:
                print("built", len(prompts), "missed", missed, flush=True)
    os.makedirs(args.dest, exist_ok=True)
    train = pa.table({"query": prompts, "completion": completions, "target": targets})
    holdout = max(len(prompts) // 20, 1)
    pq.write_table(train.slice(holdout), os.path.join(args.dest, "train.parquet"))
    pq.write_table(train.slice(0, holdout), os.path.join(args.dest, "test.parquet"))
    print("wrote", train.num_rows - holdout, "train", holdout, "heldout", "missed", missed)


if __name__ == "__main__":
    main()
