"""Turning n Countdown samples into one answer.

Every selector sees only what is available at test time: the prompt's
numbers and target, the sampled texts, and the sampler's log-probabilities.
Correctness labels are used only to score the selected sample.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass

from evaluation.countdown import evaluate_equation, extract_solution, validate_equation


def pass_at_k(n, c, k):
    """Unbiased pass@k for one prompt (Chen et al., 2021)."""
    if k > n:
        raise ValueError("k cannot exceed the number of samples")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


@dataclass
class Sample:
    key: str  # normalised answer string, "" when no <answer> span
    numbers_ok: bool
    verified: bool
    mean_logp: float


def featurise(text, mean_logp, numbers, target):
    equation = extract_solution(text)
    if equation is None:
        return Sample("", False, False, mean_logp)
    numbers_ok = validate_equation(equation, list(numbers))
    verified = False
    if numbers_ok:
        value = evaluate_equation(equation)
        verified = value is not None and abs(value - target) < 1e-5
    return Sample(re.sub(r"\s+", "", equation), numbers_ok, verified, mean_logp)


def _most_confident(samples, pool):
    return max(pool, key=lambda i: samples[i].mean_logp)


def select_max_logprob(samples):
    return _most_confident(samples, range(len(samples)))


def select_majority(samples):
    """Self-consistency over normalised answer strings; ties go to confidence."""
    counts = Counter(s.key for s in samples if s.key)
    if not counts:
        return select_max_logprob(samples)
    best = max(counts.values())
    pool = [i for i, s in enumerate(samples) if s.key and counts[s.key] == best]
    return _most_confident(samples, pool)


def select_number_check(samples):
    """Weak verifier: answers that use each number exactly once, then confidence."""
    pool = [i for i, s in enumerate(samples) if s.numbers_ok]
    return _most_confident(samples, pool or range(len(samples)))


def select_verifier(samples):
    """Full verifier built from the prompt; falls back to confidence."""
    pool = [i for i, s in enumerate(samples) if s.verified]
    return _most_confident(samples, pool or range(len(samples)))


SELECTORS = {
    "majority": select_majority,
    "max_logprob": select_max_logprob,
    "number_check": select_number_check,
    "verifier": select_verifier,
}
