# CS2470 Modal starter: GPU kernel profiling

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

## 2. Profile on a GPU

Start with A10; the first run builds the remote image and can take a few minutes.

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

## Costs and troubleshooting

Check current rates and usage:

```bash
modal billing rates
modal billing report --for today --resolution h --show-resources
modal billing summary
```

Billing reports can take a few minutes to settle and show usage **before credits**.
Granular reports require an eligible workspace plan; the dashboard is another way
to check usage. Use the App ID printed by `modal run` to attribute each experiment.
In our October 2026 tests, two full attention profiles cost roughly **$0.01–$0.11**
across A10/A100/H100/B200, including retries in some cases. This is an example,
not a price guarantee: allocation time, replay count, CPU/memory, and storage matter.

- **Clock-lock error:** this starter uses `--clock-control none`, which worked on
  our Modal GPUs. It does not attempt to change host GPU clock settings.
- **No kernels profiled:** inspect the log's available kernels and adjust the filter.
- **Counter permission error:** access depends on the host/workspace; save the log
  and check with Modal support. Installing NCU alone does not grant permissions.
- **Missing metrics:** some GPU/interconnect counters may be unavailable even when
  the kernel report completes. Read warnings before using a metric.
- **Authentication:** rerun `modal setup`; never commit tokens or credentials.

This tutorial runs **Nsight Compute**, not Nsight Systems. For CPU/GPU timelines,
use PyTorch Profiler or Nsight Systems in a separate experiment.

References: [Modal setup](https://modal.com/docs/guide/getting-started),
[GPU selection](https://modal.com/docs/guide/gpu),
[billing](https://modal.com/docs/guide/billing),
[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).
