"""The checkpointed RL loss terms must equal the reference objectives."""
import os
import sys

import torch

CODE = os.path.join(os.path.dirname(__file__), "..", "code")
sys.path.insert(0, CODE)
from countdown_objectives import token_entropy, token_kl  # noqa: E402
from rloo_trainer.rl_vllm import advantages_for, token_terms  # noqa: E402


def test_token_terms_match_reference_values_and_gradients():
    torch.manual_seed(0)
    logits = torch.randn(3, 70, 11, requires_grad=True)
    ref = torch.randn(3, 70, 11)
    input_ids = torch.randint(0, 11, (3, 71))
    weights = torch.rand(3, 70)

    logp, ent, kl = token_terms(logits, input_ids, ref, chunk=16)
    loss = ((logp + 0.3 * ent - 0.7 * kl) * weights).sum()
    loss.backward()
    grad = logits.grad.clone()

    logits2 = logits.detach().clone().requires_grad_(True)
    ref_logp = torch.log_softmax(logits2.float(), -1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    ref_ent = token_entropy(logits2)
    ref_kl = token_kl(logits2, ref)
    assert torch.allclose(logp, ref_logp, atol=1e-5)
    assert torch.allclose(ent, ref_ent, atol=1e-5)
    assert torch.allclose(kl, ref_kl, atol=1e-5)
    ((ref_logp + 0.3 * ref_ent - 0.7 * ref_kl) * weights).sum().backward()
    assert torch.allclose(grad, logits2.grad, atol=1e-5)


def test_advantage_dispatch():
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(advantages_for("rloo", r, 4), torch.tensor([1.0, -1 / 3, -1 / 3, -1 / 3]))
    assert abs(float(advantages_for("grpo", r, 4).std()) - 1.0) < 1e-3
    assert torch.allclose(advantages_for("drgrpo", r, 4), torch.tensor([0.75, -0.25, -0.25, -0.25]))
