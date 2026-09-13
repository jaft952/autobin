"""Log storage for the dashboard."""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path

import src.log_levels  # noqa: F401


class LogBuffer:
    """Thread-safe list of recent log entries."""

    def __init__(self, capacity: int = 1000):
        self._entries: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = 0

    def append(self, level: str, source: str, msg: str) -> None:
        msg = msg.rstrip()
        if not msg:
            return
        with self._lock:
            self._seq += 1
            self._entries.append({
                "seq": self._seq,
                "ts": time.strftime("%H:%M:%S"),
                "level": level,
                "source": source,
                "msg": msg,
            })

    def since(self, after_seq: int, limit: int = 500) -> list:
        with self._lock:
            return [e for e in self._entries if e["seq"] > after_seq][:limit]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class BufferLogHandler(logging.Handler):
    """Sends log messages into the LogBuffer."""

    def __init__(self, buffer: LogBuffer):
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append(record.levelname, record.name, self.format(record))
        except Exception:
            self.handleError(record)


class TeeStream:
    """Prints output and also saves each line to the LogBuffer."""

    def __init__(self, real, buffer: LogBuffer, level: str, source: str):
        self._real = real
        self._buffer = buffer
        self._level = level
        self._source = source
        self._partial = ""

    def write(self, text: str):
        self._real.write(text)
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            self._buffer.append(self._level, self._source, line)
        return len(text)

    def flush(self):
        self._real.flush()

    def isatty(self):
        return False


def setup_logging(buffer: LogBuffer, log_file: str = "logs/autobin.log",
                  capture_prints: bool = True) -> logging.Logger:
    """Connect logging and print() to the log buffer and file."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    buf_handler = BufferLogHandler(buffer)
    buf_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(buf_handler)

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(file_handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)

    if capture_prints:
        sys.stdout = TeeStream(sys.stdout, buffer, "DEBUG", "print")
        sys.stderr = TeeStream(sys.stderr, buffer, "ERROR", "stderr")

    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    return logging.getLogger("web")
