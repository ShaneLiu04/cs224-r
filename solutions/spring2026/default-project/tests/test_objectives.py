"""Closed-form checks for the Countdown SFT, IPO, and RLOO objectives."""
import os
import sys

import torch

CODE = os.path.join(os.path.dirname(__file__), "..", "code")
sys.path.insert(0, CODE)
from countdown_objectives import (
    causal_token_logprobs,
    masked_nll,
    preference_loss,
    response_mask,
    rloo_advantages,
    sequence_logprobs,
    token_entropy,
    token_kl,
)


class TinyLM(torch.nn.Module):
    """Causal stand-in: every position shares one bias over a 4-token vocabulary."""

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(4))

    def forward(self, input_ids, attention_mask, use_cache=False):
        batch, length = input_ids.shape
        logits = self.bias.view(1, 1, -1).repeat(batch, length, 1)
        return type("Out", (), {"logits": logits})()


def test_response_mask_drops_prompt_and_padding():
    is_response = torch.tensor([[0, 0, 1, 1, 1]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    mask = response_mask(is_response, attention)
    assert torch.equal(mask, torch.tensor([[0.0, 1.0, 1.0, 0.0]]))


def test_nll_ignores_prompt_tokens():
    torch.manual_seed(0)
    model = TinyLM()
    input_ids = torch.tensor([[1, 1, 2, 3]])
    attention = torch.ones_like(input_ids)
    is_response = torch.tensor([[0, 0, 1, 1]])
    _, token_logp = causal_token_logprobs(model, input_ids, attention)
    mask = response_mask(is_response, attention)
    loss = masked_nll(token_logp, mask)
    loss.backward()
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert model.bias.grad.abs().sum() > 0


def test_ipo_is_one_over_two_beta_squared_at_the_reference():
    beta = 0.1
    delta_w = torch.zeros(4)
    delta_l = torch.zeros(4)
    loss = preference_loss(delta_w, delta_l, beta, "ipo")
    assert torch.allclose(loss, torch.tensor((1.0 / (2.0 * beta)) ** 2))


def test_dpo_goes_to_zero_when_the_chosen_margin_is_large():
    loss = preference_loss(torch.tensor([80.0]), torch.tensor([0.0]), 0.1, "dpo")
    assert float(loss) < 1e-3


def test_rloo_advantages_match_leave_one_out():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.5, 0.5])
    advantages = rloo_advantages(rewards, 2)
    assert torch.allclose(advantages, torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]))
    equal = rloo_advantages(torch.tensor([0.1, 0.1, 0.1, 0.1]), 4)
    assert torch.allclose(equal, torch.zeros(4))


def test_sequence_logprob_is_the_sum_unless_averaged():
    token_logp = torch.tensor([[-1.0, -2.0, -3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    summed = sequence_logprobs(token_logp, mask, average=False)
    averaged = sequence_logprobs(token_logp, mask, average=True)
    assert torch.allclose(summed, torch.tensor([-4.0]))
    assert torch.allclose(averaged, torch.tensor([-2.0]))


def test_uniform_entropy_and_zero_kl():
    logits = torch.zeros(1, 2, 4)
    entropy = token_entropy(logits)
    expected = torch.log(torch.tensor(4.0))
    assert torch.allclose(entropy, expected.expand_as(entropy))
    assert torch.allclose(token_kl(logits, logits), torch.zeros_like(entropy))
