"""web/logbuffer.py: ring buffer feeding the dashboard log panel, fed by
Python logging, print() (via TeeStream), and direct .append() calls. Each entry gets a seq number so the frontend can poll by cursor."""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from pathlib import Path

import src.log_levels  # noqa: F401 -- registers log.success()/log.fail()


class LogBuffer:
    """Thread-safe ring buffer of structured log entries."""

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
    """Routes the logging module into the LogBuffer."""

    def __init__(self, buffer: LogBuffer):
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append(record.levelname, record.name, self.format(record))
        except Exception:
            self.handleError(record)


class TeeStream:
    """Passes writes through and collects whole lines into the LogBuffer."""

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
    """Wire root logger -> buffer + file, print() -> buffer. Returns the web logger."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # DEBUG carries per-tick chatter; INFO is for milestones.

    buf_handler = BufferLogHandler(buffer)
    buf_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(buf_handler)

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(file_handler)

    # Console output still goes to the REAL stdout (pre-tee) to avoid loops.
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)

    if capture_prints:
        # DEBUG: routine print() narration, same bucket as per-tick chatter.
        sys.stdout = TeeStream(sys.stdout, buffer, "DEBUG", "print")
        sys.stderr = TeeStream(sys.stderr, buffer, "ERROR", "stderr")

    # ultralytics/opencv at DEBUG would flood the buffer now root is DEBUG.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # werkzeug's per-request lines would flood the log panel that's polling.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    return logging.getLogger("web")
