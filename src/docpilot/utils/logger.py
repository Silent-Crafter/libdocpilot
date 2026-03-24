"""
Centralized logging configuration for DocPilot.

- Standard logs (DEBUG/INFO/WARNING/ERROR) → stdout
- Benchmark/telemetry metrics              → benchmark_metrics.jsonl (JSON Lines)

Usage:
    from docpilot.utils.logger import setup_logging, get_benchmark_logger

    setup_logging()                         # call once at application entry
    bench = get_benchmark_logger()          # per-module or shared
    bench.info(json.dumps(metrics_dict))    # one JSON object per line
"""

import logging
import sys
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ── Formatters ──────────────────────────────────────────────────────────────

class _StdoutFormatter(logging.Formatter):
    """Coloured, human-readable formatter for console output."""

    COLORS = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[34m",   # blue
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"


class _JsonLineFormatter(logging.Formatter):
    """Writes each log record as a single JSON object (one per line)."""

    def format(self, record: logging.LogRecord) -> str:
        # If the message is already a JSON string (from benchmark calls),
        # pass it through directly; otherwise wrap it.
        try:
            payload = json.loads(record.getMessage())
        except (json.JSONDecodeError, TypeError):
            payload = {"message": record.getMessage()}
        return json.dumps(payload, ensure_ascii=False)


# ── Public helpers ──────────────────────────────────────────────────────────

_LOGGING_CONFIGURED = False


def setup_logging(level: int = logging.DEBUG) -> None:
    """
    Configure the root ``docpilot`` logger hierarchy.

    * Adds a coloured ``StreamHandler`` (stdout) for all standard logs.
    * Should be called **once** at application startup.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    _LOGGING_CONFIGURED = True

    root = logging.getLogger("docpilot")
    root.setLevel(level)

    # stdout handler — human-readable
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(_StdoutFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    root.addHandler(sh)


def get_benchmark_logger(
    filepath: str = "benchmark_metrics.jsonl",
) -> logging.Logger:
    """
    Return (and lazily configure) the ``docpilot.benchmark`` logger.

    Logs emitted through this logger are written **only** to *filepath*
    as JSON Lines — they do **not** propagate to stdout.
    """
    name = "docpilot.benchmark"
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.propagate = False          # keep metrics out of stdout

        fh = logging.FileHandler(filepath, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(_JsonLineFormatter())
        logger.addHandler(fh)

    return logger


# ── Benchmark helpers ───────────────────────────────────────────────────────

class BenchmarkTimer:
    """
    Simple context-manager / manual timer for recording latency slices.

    Usage:
        timer = BenchmarkTimer()
        with timer.measure("retrieval"):
            ...
        with timer.measure("generation"):
            ...
        timer.get_latencies()
        # → {"latency_retrieval_s": 0.41, "latency_generation_s": 1.23, "latency_total_s": 1.64}
    """

    def __init__(self) -> None:
        self._start: float = time.perf_counter()
        self._slices: Dict[str, float] = {}
        self._current_label: Optional[str] = None
        self._slice_start: float = 0.0

    # context-manager style
    class _Slice:
        def __init__(self, timer: "BenchmarkTimer", label: str):
            self._timer = timer
            self._label = label

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *_: Any):
            elapsed = time.perf_counter() - self._t0
            self._timer._slices[self._label] = round(elapsed, 4)

    def measure(self, label: str) -> "_Slice":
        return self._Slice(self, label)

    def get_latencies(self) -> Dict[str, float]:
        total = round(time.perf_counter() - self._start, 4)
        out: Dict[str, float] = {}
        for label, elapsed in self._slices.items():
            out[f"latency_{label}_s"] = elapsed
        out["latency_total_s"] = total
        return out


def build_benchmark_record(
    *,
    query: str,
    rewritten_query: str = "",
    retrieved_docs: Optional[list] = None,
    retrieval_scores: Optional[list] = None,
    retrieval_threshold: float = 0.64,
    source_files: Optional[list] = None,
    retrieved_images: Optional[list] = None,
    image_retrieval_scores: Optional[list] = None,
    image_retrieval_threshold: float = 0.7,
    generated_images: Optional[list] = None,
    ground_truth_images: Optional[list] = None,
    answer: str = "",
    model: str = "",
    embed_model: str = "",
    num_hops: int = 1,
    context_chunks_used: int = 0,
    streaming: bool = False,
    latencies: Optional[Dict[str, float]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a complete benchmark metrics dict ready for JSON serialisation."""
    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "query": query,
        "rewritten_query": rewritten_query,

        "retrieved_docs": retrieved_docs or [],
        "retrieval_scores": retrieval_scores or [],
        "num_docs_retrieved": len(retrieved_docs or []),
        "num_docs_above_threshold": sum(
            1 for s in (retrieval_scores or []) if s >= retrieval_threshold
        ),
        "retrieval_threshold": retrieval_threshold,
        "source_files": source_files or [],

        "retrieved_images": retrieved_images or [],
        "image_retrieval_scores": image_retrieval_scores or [],
        "image_retrieval_threshold": image_retrieval_threshold,
        "generated_images": generated_images or [],
        "ground_truth_images": ground_truth_images or [],

        "answer": answer,
        "answer_length": len(answer),

        "model": model,
        "embed_model": embed_model,
        "num_hops": num_hops,
        "context_chunks_used": context_chunks_used,
        "streaming": streaming,
    }

    if latencies:
        record.update(latencies)

    if extra:
        record.update(extra)

    return record
