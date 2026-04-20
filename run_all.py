from __future__ import annotations

import atexit
import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full SGD vs AdamW experiment")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 17])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--log-file", type=Path, default=Path("execution.log"))
    parser.add_argument("--lock-file", type=Path, default=Path(".run_all.lock"))
    return parser.parse_args()


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("run_all")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock(lock_file: Path, logger: logging.Logger) -> None:
    if lock_file.exists():
        text = lock_file.read_text(encoding="utf-8").strip()
        if text.isdigit() and _pid_is_running(int(text)):
            raise RuntimeError(
                f"Another run_all instance is already running (pid={text}). "
                "Stop it first or remove stale lock file."
            )
        logger.warning("Removing stale lock file: %s", lock_file)
        lock_file.unlink(missing_ok=True)

    lock_file.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(lambda: lock_file.unlink(missing_ok=True))


def run_command_stream(cmd: list[str], logger: logging.Logger) -> tuple[int, list[str]]:
    # Stream subprocess output to logger line-by-line so long runs
    # show continuous progress in execution logs.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        stripped = line.rstrip("\n")
        lines.append(stripped)
        logger.info("%s", stripped)
    code = proc.wait()
    return code, lines


def run_training(
    optimizer: str,
    seed: int,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> bool:
    base_cmd = [
        sys.executable,
        "-u",
        "train.py",
        "--optimizer",
        optimizer,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--results-dir",
        str(args.results_dir),
        "--data-dir",
        str(args.data_dir),
    ]

    cpu_forced = False
    for attempt in range(args.max_retries + 1):
        cmd = list(base_cmd)
        if cpu_forced:
            cmd.append("--no-cuda")

        logger.info(
            "Starting %s seed %s (attempt %s/%s)%s",
            optimizer.upper(),
            seed,
            attempt + 1,
            args.max_retries + 1,
            " [CPU mode]" if cpu_forced else "",
        )
        started = time.time()
        returncode, lines = run_command_stream(cmd, logger=logger)
        elapsed = time.time() - started

        if returncode == 0:
            acc = parse_final_accuracy(lines)
            logger.info(
                "Completed %s seed %s in %.1fs (final/best accuracy: %s)",
                optimizer.upper(),
                seed,
                elapsed,
                acc,
            )
            return True

        merged_err = "\n".join(lines).lower()
        logger.error(
            "Failed %s seed %s in %.1fs (code=%s)",
            optimizer.upper(),
            seed,
            elapsed,
            returncode,
        )

        if "out of memory" in merged_err and not cpu_forced:
            logger.warning("CUDA out of memory detected. Switching to CPU retries.")
            cpu_forced = True

    logger.error("Giving up: %s seed %s", optimizer.upper(), seed)
    return False


def parse_final_accuracy(lines: list[str]) -> str:
    for line in reversed(lines):
        if "best_val_accuracy=" in line:
            return line.split("best_val_accuracy=")[-1].strip()
        if "val_accuracy=" in line:
            return line.split("val_accuracy=")[-1].strip()
    return "n/a"


def run_analysis(args: argparse.Namespace, logger: logging.Logger) -> bool:
    cmd = [
        sys.executable,
        "-u",
        "analyze_results.py",
        "--results-dir",
        str(args.results_dir),
        "--output-dir",
        str(args.results_dir),
        "--data-dir",
        str(args.data_dir),
    ]
    logger.info("Starting analysis")
    returncode, _ = run_command_stream(cmd, logger=logger)
    if returncode != 0:
        logger.error("analysis failed with code=%s", returncode)
        return False
    logger.info("analysis completed")
    return True


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.log_file)
    try:
        acquire_lock(args.lock_file, logger=logger)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    total = 0
    success = 0

    for optimizer in ["sgd", "adamw"]:
        for seed in args.seeds:
            total += 1
            ok = run_training(optimizer=optimizer, seed=seed, args=args, logger=logger)
            if ok:
                success += 1

    logger.info("Training summary: %s/%s successful runs", success, total)

    analysis_ok = run_analysis(args, logger)
    if not analysis_ok:
        logger.warning("Analysis step failed. Check logs for details.")

    logger.info("Pipeline finished")
    return 0 if success > 0 and analysis_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
