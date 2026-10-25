"""Collect GPU power on Modal: modal run power.py --method pynvml --gpu A10."""
import csv
import io
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import modal

app = modal.App("cs2470-gpu-power")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.14.1", "nvidia-ml-py"
)
GPUS = ("A10", "L4", "A100-40GB", "A100-80GB", "H100", "H100!", "B200")


def log_nvml(stop, ready):
    """Sample in a background thread while the main thread runs the workload."""
    import pynvml

    pynvml.nvmlInit()
    try:
        # This example requests one GPU. UUID identifies it in the saved trace.
        gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["elapsed_s", "uuid", "power_W", "gpu_util_percent"])
        started = time.perf_counter()
        while not stop.is_set():
            writer.writerow([
                time.perf_counter() - started,
                pynvml.nvmlDeviceGetUUID(gpu),
                pynvml.nvmlDeviceGetPowerUsage(gpu) / 1000,  # mW -> W
                pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu,
            ])
            ready.set()
            stop.wait(0.1)
        return output.getvalue()
    finally:
        ready.set()  # Also unblock the caller if sampling fails.
        pynvml.nvmlShutdown()


@app.function(image=image, gpu="A10", cpu=2, memory=8192, timeout=180)
def measure(method: str, seconds: float):
    import torch

    print("GPU:", torch.cuda.get_device_name(0))
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    out = torch.empty_like(a)

    def workload(duration):
        # Replace this loop with your model or kernel; keep inputs allocated above.
        deadline = time.perf_counter() + duration
        while time.perf_counter() < deadline:
            for _ in range(10):
                torch.mm(a, b, out=out)
            torch.cuda.synchronize()  # Finish queued work before checking time.

    workload(2)  # Warm up before starting the logger.
    if method == "pynvml":
        stop, ready = threading.Event(), threading.Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(log_nvml, stop, ready)
            try:
                if not ready.wait(10):
                    raise RuntimeError("NVML logger did not start")
                if future.done():
                    future.result()  # Surface telemetry errors before running.
                workload(seconds)
            finally:
                stop.set()
            return future.result()

    # Run ONE persistent nvidia-smi process, not a new process for each sample.
    with tempfile.TemporaryFile(mode="w+") as output:
        process = subprocess.Popen([
            "nvidia-smi", "--id=0",
            "--query-gpu=timestamp,uuid,power.draw,utilization.gpu",
            "--format=csv,nounits", "-lms", "100",
        ], stdout=output, stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(0.3)  # Give the logger time to initialize.
            if process.poll() is not None:
                raise RuntimeError(process.communicate()[1])
            workload(seconds)
            if process.poll() is not None:
                raise RuntimeError(process.communicate()[1])
        finally:
            process.terminate()
            process.communicate(timeout=10)
        output.seek(0)
        return output.read()


@app.local_entrypoint()
def main(gpu: str = "A10", method: str = "pynvml", seconds: float = 10):
    if gpu not in GPUS or method not in ("pynvml", "smi"):
        raise ValueError(f"Choose GPU from {GPUS} and method pynvml or smi")
    if not 1 <= seconds <= 60:
        raise ValueError("Use --seconds between 1 and 60")
    trace = measure.with_options(gpu=gpu).remote(method, seconds)
    folder = Path("artifacts") / f"power-{gpu.lower().replace('!', '')}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    path = folder / f"{method}.csv"
    path.write_text(trace)
    print(f"Saved GPU power trace to {path}")
