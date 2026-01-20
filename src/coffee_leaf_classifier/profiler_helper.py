from pathlib import Path

import torch
import torch.profiler as profiler
from loguru import logger

def make_pl_profiler():
    """
    Returns a PyTorch Lightning PyTorchProfiler configured to write TensorBoard traces
    and a Chrome trace.
    """
    try:
        from pytorch_lightning.profilers import PyTorchProfiler
    except Exception as e:
        raise RuntimeError(
            "Could not import PyTorchProfiler from pytorch_lightning.profilers. "
            "Paste the error and your pytorch_lightning version."
        ) from e

    trace_dir = Path("runs") / "profiler"
    trace_dir.mkdir(parents=True, exist_ok=True)

    activities = [profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(profiler.ProfilerActivity.CUDA)

    sched = profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

    tb_handler = profiler.tensorboard_trace_handler(str(trace_dir))

    return PyTorchProfiler(
        dirpath=str(trace_dir),
        filename="pl_profile",
        export_to_chrome=True,
        activities=activities,
        schedule=sched,
        on_trace_ready=tb_handler,
        record_shapes=True,
        with_stack=True,
    )


def generate_profiling_report() -> None:
    """
    Generate a summary profiling report from the saved profiling traces.
    Saves the report to runs/profiler/profiling_report.txt
    """
    trace_dir = Path("runs") / "profiler"
    
    if not trace_dir.exists():
        logger.warning("Profiling directory does not exist. No report to generate.")
        return
    
    report_path = trace_dir / "profiling_report.txt"
    
    try:
        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("PYTORCH PROFILER REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # List trace files
            trace_files = list(trace_dir.glob("*.json"))
            chrome_traces = list(trace_dir.glob("*.json.gz"))
            
            f.write("Generated Trace Files:\n")
            f.write("-" * 80 + "\n")
            
            if trace_files:
                f.write("\nJSON Traces:\n")
                for file in trace_files:
                    size_mb = file.stat().st_size / (1024 * 1024)
                    f.write(f"  - {file.name} ({size_mb:.2f} MB)\n")
            
            if chrome_traces:
                f.write("\nChrome Traces (gzip):\n")
                for file in chrome_traces:
                    size_mb = file.stat().st_size / (1024 * 1024)
                    f.write(f"  - {file.name} ({size_mb:.2f} MB)\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("\nHow to Analyze:\n")
            f.write("-" * 80 + "\n")
            f.write("1. Chrome Traces (Chrome DevTools):\n")
            f.write("   - Open Chrome: chrome://tracing\n")
            f.write("   - Load *.json.gz files to visualize timeline\n\n")
            f.write("2. TensorBoard:\n")
            f.write(f"   - tensorboard --logdir={trace_dir}\n")
            f.write("   - Open http://localhost:6006\n\n")
            f.write("3. Key Metrics to Look For:\n")
            f.write("   - Wall clock time (total training duration)\n")
            f.write("   - CPU/GPU utilization percentage\n")
            f.write("   - Memory peak usage\n")
            f.write("   - Kernel execution time\n")
            f.write("   - Data loading vs compute ratio\n")
            
        logger.info(f"Profiling report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate profiling report: {e}")
