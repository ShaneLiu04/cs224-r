"""Ray actor that serves batched generation for RLOO training.

This process owns a vLLM engine and is restarted when checkpoints change.
"""

import warnings
warnings.filterwarnings("ignore")

import ray
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

@ray.remote(num_gpus=1)
class SamplingWorker:
    """GPU worker responsible for policy rollouts (text sampling)."""
    def __init__(
        self, 
        model_path, 
        max_model_len=2048, 
        gpu_memory_utilization=0.9, 
        max_num_batched_tokens=8192, 
        enable_chunked_prefill=True, 
        max_num_seqs=64,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        max_tokens=1024,
        group_size=16
    ):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_seqs = max_num_seqs
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.group_size = group_size

    def load_checkpoint(self):
        """(Re)load tokenizer + vLLM engine for current model path."""
        self.tear_down()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        effective_max_model_len = self.max_model_len
        tokenizer_max_length = getattr(self.tokenizer, "model_max_length", None)
        if (
            effective_max_model_len is not None
            and isinstance(tokenizer_max_length, int)
            and 0 < tokenizer_max_length < 1_000_000
        ):
            effective_max_model_len = min(effective_max_model_len, tokenizer_max_length)
        effective_max_num_batched_tokens = self.max_num_batched_tokens
        if (
            effective_max_model_len is not None
            and effective_max_num_batched_tokens is not None
            and effective_max_num_batched_tokens < effective_max_model_len
        ):
            effective_max_num_batched_tokens = effective_max_model_len

        # Build vLLM config once so callers can hot-swap model paths cleanly.
        llm_kwargs = {
            "model": self.model_path,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_batched_tokens": effective_max_num_batched_tokens,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "max_num_seqs": self.max_num_seqs,
        }
        if effective_max_model_len is not None:
            llm_kwargs["max_model_len"] = effective_max_model_len

        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            n=self.group_size,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            logprobs=1
        )

    def update_model_path(self, model_path):
        """Switch to a new checkpoint and reload generation engine."""
        self.model_path = model_path
        self.load_checkpoint()

    def tear_down(self):
        """Release GPU memory and distributed state before reload/exit."""
        import gc
        import torch
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if hasattr(self, 'llm'):
            # vLLM requires explicit cleanup of distributed resources
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except Exception:
                pass
            del self.llm
        if hasattr(self, 'sampling_params'):
            del self.sampling_params
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @staticmethod
    def _extract_sequence_logprob(output) -> float:
        """Best-effort extraction of sequence logprob across vLLM versions."""
        if hasattr(output, "cumulative_logprob") and output.cumulative_logprob is not None:
            return float(output.cumulative_logprob)
        if hasattr(output, "logprob") and output.logprob is not None:
            return float(output.logprob)

        # Fallback path: reconstruct from per-token logprobs if cumulative logprob
        # is not present in this vLLM version.
        token_ids = getattr(output, "token_ids", None)
        token_logprobs = getattr(output, "logprobs", None)
        if token_ids is None or token_logprobs is None:
            raise RuntimeError(
                "Could not extract sequence logprob from vLLM output: missing cumulative_logprob/logprob and token-level logprobs."
            )

        seq_logprob = 0.0
        for token_id, token_topk in zip(token_ids, token_logprobs):
            if token_topk is None:
                continue

            token_entry = None
            if isinstance(token_topk, dict):
                token_entry = token_topk.get(token_id)
                if token_entry is None and len(token_topk) == 1:
                    token_entry = next(iter(token_topk.values()))
            else:
                token_entry = token_topk

            if token_entry is None:
                continue

            if hasattr(token_entry, "logprob"):
                seq_logprob += float(token_entry.logprob)
            else:
                seq_logprob += float(token_entry)

        return float(seq_logprob)

    def generate(self, prompts: list[str]) -> tuple[list[list[str]], list[list[float]]]:
        """Sample `group_size` responses per prompt and return sequence logprobs."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        all_responses = []
        all_logprobs = []
        for output in outputs:
            curr_responses = []
            curr_logprobs = []
            for o in output.outputs:
                curr_responses.append(o.text)
                curr_logprobs.append(self._extract_sequence_logprob(o))
            all_responses.append(curr_responses)
            all_logprobs.append(curr_logprobs)
        return all_responses, all_logprobs
