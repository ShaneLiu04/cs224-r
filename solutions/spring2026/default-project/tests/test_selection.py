"""Checks for the answer selectors used to turn pass@n into pass@1."""
import os
import random
import sys

CODE = os.path.join(os.path.dirname(__file__), "..", "code")
sys.path.insert(0, CODE)
from evaluation.countdown_eval_vllm import score_prompt  # noqa: E402
from evaluation.selection import (  # noqa: E402
    featurise, pass_at_k, select_majority, select_max_logprob,
    select_number_check, select_verifier,
)

NUMS, TARGET = [3, 5, 7], 22


def make(texts, logps):
    return [featurise(t, lp, NUMS, TARGET) for t, lp in zip(texts, logps)]


def test_featurise_flags():
    good = featurise("<answer> 3 * 5 + 7 </answer>", -0.1, NUMS, TARGET)
    wrong_value = featurise("<answer>3 + 5 + 7</answer>", -0.1, NUMS, TARGET)
    reused = featurise("<answer>7 + 7 + 3 + 5</answer>", -0.1, NUMS, TARGET)
    missing = featurise("I think 22", -0.1, NUMS, TARGET)
    assert good.verified and good.numbers_ok and good.key == "3*5+7"
    assert wrong_value.numbers_ok and not wrong_value.verified
    assert not reused.numbers_ok and not reused.verified
    assert missing.key == "" and not missing.numbers_ok


def test_majority_prefers_most_common_answer_string():
    samples = make(
        ["<answer>3+5+7</answer>", "<answer> 3 + 5 + 7 </answer>", "<answer>3*5+7</answer>"],
        [-1.0, -1.0, -0.1],
    )
    assert select_majority(samples) in (0, 1)
    assert select_max_logprob(samples) == 2


def test_verifier_picks_a_correct_sample_whenever_one_exists():
    samples = make(
        ["<answer>3+5+7</answer>", "<answer>7*3+5</answer>", "<answer>3*5+7</answer>"],
        [-0.1, -0.2, -3.0],
    )
    assert select_verifier(samples) == 2
    assert select_number_check(samples) == 0


def test_verifier_at_full_budget_equals_pass_at_n():
    rng = random.Random(0)
    samples = make(
        ["<answer>3+5+7</answer>"] * 7 + ["<answer>3*5+7</answer>"],
        [-0.1] * 7 + [-2.0],
    )
    item = score_prompt(samples, [1, 2, 4, 8], rng, draws=256)
    assert item["verifier@8"] == 1.0 == item["pass@8"]
    assert abs(item["pass@1"] - 1 / 8) < 1e-9
    assert abs(item["verifier@4"] - pass_at_k(8, 1, 4)) < 0.08
    assert item["max_logprob@8"] == 0.0


def test_pass_at_k_edge_cases():
    assert pass_at_k(16, 0, 8) == 0.0
    assert pass_at_k(16, 16, 1) == 1.0
    assert abs(pass_at_k(4, 1, 1) - 0.25) < 1e-12
