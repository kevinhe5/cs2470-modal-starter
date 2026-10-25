# Quickstart: GPU power on Modal

Run the [environment setup in the README](README.md#1-set-up-your-environment)
first. Only Modal is installed on your laptop; `power.py` installs PyTorch and
NVIDIA's Python bindings in the remote image. No local GPU is needed.

## 1. Collect a power trace

From your local terminal, in this repository:

```bash
modal run power.py --method smi --gpu A10
modal run power.py --method pynvml --gpu A10
```

Choose either method. Each command rents one GPU, warms up a small matrix
multiplication workload, samples power while it runs for about 10 seconds, then
saves a CSV to `artifacts/power-<gpu>-<id>/` on your laptop. The first run may take
longer to build the image. These runs use your workspace's Modal credits.

To change the GPU or workload duration:

```bash
modal run power.py --method pynvml --gpu A100-80GB --seconds 20
modal run power.py --method pynvml --gpu B200 --seconds 20
```

Other choices include L4, A100-40GB, and `'H100!'` (the `!` prevents an automatic
upgrade to H200).

## 2. What the two methods do

**NVML** stands for NVIDIA Management Library: the driver's API for reading GPU
power, temperature, utilization, and other telemetry. `nvidia-smi` is a command-line
interface built on NVML. Python calls the same library through `pynvml`.

### nvidia-smi: simplest CSV logger

Inside the remote GPU container, the script starts this command alongside the
workload and stops it afterward:

```bash
nvidia-smi --id=0 \
  --query-gpu=timestamp,uuid,power.draw,utilization.gpu \
  --format=csv,nounits -lms 100 > power.csv
```

The columns are time, GPU UUID, watts, and GPU utilization percent. `-lms 100`
requests a sample every 100 milliseconds. If running this command manually in a
GPU shell, stop it with Ctrl+C. Running it on your laptop does not monitor Modal.

### pynvml: convenient inside Python

The package is named **`nvidia-ml-py`**, but its import name is **`pynvml`**.
The remote image installs it with `.pip_install("nvidia-ml-py")`.
A single reading inside a remote function looks like:

```python
import pynvml

pynvml.nvmlInit()
try:
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)  # Our function requests one GPU.
    watts = pynvml.nvmlDeviceGetPowerUsage(gpu) / 1000  # milliwatts to watts
    print(f"GPU power: {watts:.1f} W")
finally:
    pynvml.nvmlShutdown()
```

In [power.py](power.py), a background thread repeats this reading every 100 ms
while the main thread runs matrix multiplications. The CSV contains elapsed
seconds, GPU UUID, watts, and utilization percent. Replace the `workload()` loop
with your project's computation; keep initialization and warm-up outside logging.

## 3. Interpret the measurements

- Power is in **watts**; energy is in **joules**. Estimate energy by integrating
  power over time, using actual timestamps rather than assuming exact intervals:
  `energy_J ≈ sum((P[i] + P[i+1]) / 2 * (t[i+1] - t[i]))`.
- Polling every 100 ms does not guarantee fresh measurements every 100 ms.
  Hardware updates and averaging windows vary by GPU; Python does not improve
  the sensor's resolution over `nvidia-smi`.
- These readings describe GPU device power, not per-process power or total server
  power. They include the GPU's baseline consumption.
- For tiny kernels, repeat the operation for several seconds. Telemetry cannot
  isolate the power of a single microsecond-long kernel execution.
- Measure power separately from NCU: its replay passes alter execution. This
  example does not run NCU.
- Compare energy for the **same amount of work** across GPUs. This demo runs for
  a fixed time, so faster GPUs perform more multiplications; watts alone do not
  establish which GPU is more energy efficient.

References: [NVIDIA nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/),
[NVIDIA Python bindings](https://pypi.org/project/nvidia-ml-py/).
