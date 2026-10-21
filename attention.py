"""GPU workload launched by profile_gpu.py inside the remote container.

Synthetic Llama 3.2 1B attention: batch=1, 32 query heads, 8 KV heads,
head dimension=64, BF16. Shapes AND strides match our model trace.
No model weights are downloaded. Replace make_workload() for your project.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

CASES = {
    "prefill": {
        "q_shape": (1, 32, 128, 64), "q_stride": (262144, 64, 2048, 1),
        "kv_shape": (1, 8, 128, 64), "kv_stride": (65536, 8192, 64, 1),
        "causal": True,
    },
    "decode": {
        "q_shape": (1, 32, 1, 64), "q_stride": (2048, 64, 64, 1),
        "kv_shape": (1, 8, 129, 64), "kv_stride": (66048, 8256, 64, 1),
        # The cached keys contain only past/current tokens, so all are visible.
        "causal": False,
    },
}


def make_workload(phase: str):
    """Return a zero-argument GPU operation and its correctness metadata."""
    spec = CASES[phase]

    def tensor(shape, stride):
        return torch.empty_strided(shape, stride, device="cuda", dtype=torch.bfloat16).normal_()

    q = tensor(spec["q_shape"], spec["q_stride"])
    k = tensor(spec["kv_shape"], spec["kv_stride"])
    v = tensor(spec["kv_shape"], spec["kv_stride"])

    def workload():
        # Let PyTorch select the backend for the GPU. This may be PyTorch
        # FlashAttention or a cuDNN-generated kernel, depending on hardware.
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=spec["causal"],
            scale=0.125, enable_gqa=True,
        )

    # Independent FP32 reference: expand 8 KV heads to match 32 query heads.
    # This is deliberately outside the NCU capture region.
    scores = (q.float() @ k.float().repeat_interleave(4, dim=1).transpose(-2, -1)) * 0.125
    if spec["causal"]:
        mask = torch.ones(q.shape[-2], k.shape[-2], device="cuda", dtype=torch.bool).tril()
        scores.masked_fill_(~mask, float("-inf"))
    reference = scores.softmax(-1) @ v.float().repeat_interleave(4, dim=1)
    actual = workload()
    torch.testing.assert_close(actual.float(), reference, atol=0.02, rtol=0.02)
    return workload, {
        "phase": phase, "spec": spec, "dtype": "bfloat16", "scale": 0.125,
        "enable_gqa": True, "correctness_passed": True,
        "max_abs_error_vs_fp32": (actual.float() - reference).abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=CASES, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    workload, metadata = make_workload(args.phase)

    # Initialize libraries and warm the operation before starting collection.
    for _ in range(10):
        workload()
    torch.cuda.synchronize()
    metadata.update({
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "device": str(torch.cuda.get_device_properties(0)),
        "warmup_iterations": 10,
    })
    args.metadata.write_text(json.dumps(metadata, indent=2))

    # profile_gpu.py passes --profile-from-start off. Only this operation is
    # captured. NCU replays the selected kernel to gather its hardware counters.
    torch.cuda.cudart().cudaProfilerStart()
    output = workload()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    del output


if __name__ == "__main__":
    main()
