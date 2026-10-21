"""Run from your laptop: modal run profile_gpu.py --gpu A10

The remote worker profiles attention, persists the reports in a Modal Volume,
and the local entrypoint downloads the artifacts. No local CUDA is required.
"""
import csv
import datetime
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import modal

app = modal.App("cs2470-ncu")
reports = modal.Volume.from_name("cs2470-ncu-reports", create_if_missing=True)
GPUS = ("A10", "L4", "A100-40GB", "A100-80GB", "H100", "H100!", "B200")
NCU_VERSION = "2026.3.1.2"


def install_ncu():
    """Install the pinned Linux CLI during image build, before renting a GPU."""
    import tarfile
    import urllib.request

    archive = f"nsight_compute-linux-x86_64-{NCU_VERSION}-archive.tar.xz"
    url = "https://developer.download.nvidia.com/compute/cuda/redist/nsight_compute/linux-x86_64/" + archive
    urllib.request.urlretrieve(url, "/tmp/ncu.tar.xz")
    with tarfile.open("/tmp/ncu.tar.xz") as tar:
        tar.extractall("/opt/ncu", filter="data")
    Path("/tmp/ncu.tar.xz").unlink()


# Local requirements only contain Modal; these packages are installed remotely.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1")
    .run_function(install_ncu)
    .add_local_file(Path(__file__).parent / "attention.py", "/root/attention.py")
)


def execute(command, log_path: Path, timeout: int):
    """Save diagnostics even when NCU fails or exceeds its time limit."""
    started = time.perf_counter()
    with log_path.open("w") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            return {"returncode": result.returncode, "seconds": time.perf_counter() - started}
        except subprocess.TimeoutExpired:
            log.write(f"\nTimed out after {timeout} seconds.\n")
            return {"returncode": None, "timeout": True, "seconds": time.perf_counter() - started}


@app.function(image=image, gpu="A10", cpu=2, memory=8192,
              volumes={"/reports": reports}, timeout=900)
def collect(gpu: str, phase: str, metric_set: str, kernel_filter: str):
    # UUID prevents collisions when students share a workspace or run in parallel.
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{gpu.lower().replace('!', '')}-{stamp}-{uuid.uuid4().hex[:8]}"
    out = Path("/reports") / run_id
    out.mkdir(parents=True)
    ncu = str(next(Path("/opt/ncu").glob("*/ncu")))
    summary = {"run_id": run_id, "requested_gpu": gpu, "metric_set": metric_set, "phases": {}}
    execute(["nvidia-smi"], out / "nvidia-smi.txt", 15)
    execute([ncu, "--version"], out / "ncu-version.txt", 15)
    # Preserve the exact workload used alongside each report.
    (out / "attention.py").write_text(Path("/root/attention.py").read_text())

    for name in (("prefill", "decode") if phase == "both" else (phase,)):
        print(f"Profiling {name} on {gpu} ({metric_set} metrics)", flush=True)
        command = [
            ncu,
            "--clock-control", "none",  # Clock locking failed on our Modal workers.
            "--profile-from-start", "off",  # Workload marks its capture region.
            "--set", metric_set,
            "--kernel-name", f"regex:{kernel_filter}",
            "--launch-count", "1",  # One kernel, potentially many replay passes.
            "--export", str(out / name),
            sys.executable, "/root/attention.py", "--phase", name,
            "--metadata", str(out / f"{name}-metadata.json"),
        ]
        status = execute(command, out / f"{name}.log", 240)
        status["command"] = command
        report_files = list(out.glob(f"{name}.ncu-rep*"))
        log = (out / f"{name}.log").read_text()
        # NCU can return zero even when a filter matches no kernels!
        status["success"] = (status["returncode"] == 0 and bool(report_files)
                             and "No kernels were profiled" not in log)
        status["reports"] = [p.name for p in report_files]
        if status["success"]:
            for report in report_files:
                for extension, flags in (("txt", []), ("csv", ["--csv"])):
                    result = execute(
                        [ncu, "--import", str(report), "--page", "details", *flags],
                        out / f"{name}-metrics.{extension}", 60,
                    )
                    if result["returncode"] != 0:
                        status["success"] = False
                        status["export_error"] = f"Could not export {extension} metrics"
            if status["success"]:
                rows = list(csv.DictReader(io.StringIO((out / f"{name}-metrics.csv").read_text())))
                # An imported report must contain actual measured metric rows.
                if not rows:
                    status["success"] = False
                    status["export_error"] = "The report contained no metrics"
                else:
                    status["kernel"] = rows[0]["Kernel Name"]
                    wanted = {"Duration", "DRAM Throughput", "Compute (SM) Throughput", "Achieved Occupancy"}
                    status["metrics"] = {r["Metric Name"]: {"value": r["Metric Value"], "unit": r["Metric Unit"]}
                                         for r in rows if r["Metric Name"] in wanted}
        summary["phases"][name] = status
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        # Keep completed work if a later phase is interrupted.
        reports.commit()

    # JSON avoids returning GPU-library objects that cannot unpickle locally.
    return json.dumps({"summary": summary, "files": [p.name for p in out.iterdir()]})


@app.local_entrypoint()
def main(gpu: str = "A10", phase: str = "both", metric_set: str = "full",
         kernel_filter: str = ".*(flash|sdpa).*", output_dir: str = "artifacts"):
    if gpu not in GPUS:
        raise ValueError(f"Choose a GPU from {GPUS}")
    if phase not in ("both", "prefill", "decode"):
        raise ValueError("--phase must be both, prefill, or decode")
    if metric_set not in ("basic", "full"):
        raise ValueError("--metric-set must be basic or full")

    # with_options changes the remote hardware for this invocation.
    result = json.loads(collect.with_options(gpu=gpu).remote(gpu, phase, metric_set, kernel_filter))
    summary = result["summary"]
    directory = Path(output_dir) / summary["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    for name in result["files"]:
        with (directory / name).open("wb") as handle:
            for chunk in reports.read_file(f"/{summary['run_id']}/{name}"):
                handle.write(chunk)
    print(f"Reports downloaded to {directory.resolve()}")
    for name, status in summary["phases"].items():
        print(name, "OK" if status["success"] else "FAILED", status.get("metrics", {}))
    if not all(status["success"] for status in summary["phases"].values()):
        raise RuntimeError(f"Profiling was incomplete; inspect the logs in {directory}")
