# CS2470 Modal starter: GPU kernel profiling

For access to the academic cluster, see the [HUIT OOD SSH setup guide](HUIT_OOD_SSH_SETUP.md).

Run Python on a remote GPU, collect NVIDIA Nsight Compute (NCU) hardware counters,
and download the reports to your laptop. You do **not** need a local GPU, CUDA,
PyTorch, or Hugging Face account for this tutorial.

The example profiles **synthetic attention matching Llama 3.2 1B**: BF16, batch 1,
32 query heads, 8 KV heads, and head dimension 64. Prefill uses 128 tokens; decode
uses one query token and 129 cached tokens. It does not run the full model.

## 1. Set up your environment

Use Python 3.10+ (3.12 recommended). On macOS/Linux:

```bash
git clone https://github.com/kevinhe5/cs2470-modal-starter.git
cd cs2470-modal-starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
modal setup
modal token info
modal run hello.py
```

Complete the browser sign-in with your Modal account. Use the workspace assigned
to your project and confirm its credits under **Settings → Usage & Billing**.
GPU access requires the workspace's billing/payment setup. The hello example
should print `The square is 1764`.

On Windows PowerShell, create the environment with `py -m venv .venv` and activate
it with `.venv\Scripts\Activate.ps1`. The remaining commands are the same.

## 2. Profile on a GPU with NVIDIA Nsight Compute (NCU)

These commands run **NVIDIA Nsight Compute (NCU)** to collect GPU kernel hardware
counters. Start with A10; the first run builds the remote image and can take a few
minutes.

```bash
modal run profile_gpu.py --gpu A10
```

Choose another GPU, or profile just one phase:

```bash
modal run profile_gpu.py --gpu A100-80GB
modal run profile_gpu.py --gpu 'H100!'
modal run profile_gpu.py --gpu B200 --phase decode
modal run profile_gpu.py --gpu A10 --phase prefill --metric-set basic
```

`H100!` disables Modal's automatic H100-to-H200 upgrade. The default metric set is
`full`; `basic` collects fewer metrics. GPU selection is also supported for L4
and A100-40GB. Each invocation profiles **one kernel per selected phase**.

## 3. Read your results

Results are downloaded to `artifacts/<gpu>-<timestamp>-<id>/` and persisted in the
Modal Volume `cs2470-ncu-reports`:

| File | Purpose |
| --- | --- |
| `prefill-metrics.txt`, `decode-metrics.txt` | Readable counters and NCU diagnostics |
| `*-metrics.csv` | Metrics for your own plots/tables |
| `*.ncu-repz` (or `*.ncu-rep`) | Open in a compatible Nsight Compute desktop GUI |
| `*-metadata.json` | Actual GPU, shapes, strides, versions, correctness check |
| `*.log`, `summary.json` | Completion status, command, and diagnostics |
| `attention.py` | Exact workload source used in this run |

Look at kernel duration, DRAM/SM throughput, achieved occupancy, register usage,
and warp stalls. **100% GPU utilization in Modal's dashboard does not mean peak
compute throughput.** NCU provides the more detailed measurements.

These are instrumented kernel measurements with **unlocked clocks**, not an
end-to-end inference benchmark. NCU may replay a kernel dozens of times. Measure
application latency separately without NCU. PyTorch can select different attention
kernels on different GPUs, so check the kernel name before comparing results.

For GPU power measurements with `nvidia-smi` or Python (`pynvml`), see the
[GPU power quickstart](GPU_POWER.md). It includes a runnable Modal example and
CSV downloads.

## 4. Adapt it for your project

- `hello.py`: minimal example of a local entrypoint calling a remote function.
- `attention.py`: replace `make_workload()` with your CUDA/PyTorch/Triton operation
  and a correctness check. Keep allocation and warm-up outside the capture region.
- `profile_gpu.py`: defines the remote image, GPU resources, NCU invocation, and report
  downloads. Add remote dependencies to `image` here; local requirements need only Modal.

After changing the workload, choose a matching kernel-name filter:

```bash
modal run profile_gpu.py --gpu A10 --phase prefill --kernel-filter '.*my_kernel.*'
```

The default filter matches both PyTorch FlashAttention and cuDNN attention names.
A filter that matches nothing can make NCU exit **zero** without a report; this
starter detects that and fails after downloading the logs. For multi-kernel
operations, adjust `--launch-count` in `profile_gpu.py` (currently one).

References: [Modal setup](https://modal.com/docs/guide/getting-started),
[GPU selection](https://modal.com/docs/guide/gpu),
[billing](https://modal.com/docs/guide/billing),
[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).
