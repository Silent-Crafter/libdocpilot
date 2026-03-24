"""
Centralized logging configuration for DocPilot.

- Standard logs (DEBUG/INFO/WARNING/ERROR) → stdout
"""

import logging
import sys

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

