"""
web/logbuffer.py

Log system backing the dashboard: one in-memory ring buffer that everything
funnels into, so the web UI can poll increments with a cursor.

Three inflows:
    1. Python logging      — BufferLogHandler attached to the root logger
    2. print() output      — TeeStream wraps sys.stdout/stderr; the existing
                             modules narrate via print ([GraspPlanner] ...,
                             [Actuator] ...), and those lines ARE the useful
                             robot log, so they're captured too
    3. direct .append()    — the runtime/server log their own events

Every entry gets a monotonically increasing `seq`, so the frontend asks
"give me everything after seq N" and never misses or duplicates lines.
A plain FileHandler keeps a persistent copy on disk.
"""
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
    """Wraps a real stream: passes writes through AND collects whole lines
    into the LogBuffer. Replaces sys.stdout/sys.stderr."""

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
    """Wire everything: root logger -> buffer + file, print() -> buffer.
    Returns a logger for the web package's own messages."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # DEBUG carries the per-tick "which layer
    # is running" chatter (runtime.py logs it via self.log.debug); INFO
    # stays for one-off milestone/record events (startup, mode switches,
    # manual actions, settings changes).

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
        # DEBUG: routine print() narration ("[arm] resuming...", model/
        # camera startup lines) — same bucket as the per-tick layer chatter,
        # tuckable away from the INFO/SUCCESS/WARNING/FAIL/ERROR event log.
        sys.stdout = TeeStream(sys.stdout, buffer, "DEBUG", "print")
        sys.stderr = TeeStream(sys.stderr, buffer, "ERROR", "stderr")

    # ultralytics/opencv logging at DEBUG would otherwise flood the buffer
    # now that root is DEBUG-level.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # The dashboard polls every second — werkzeug's per-request access lines
    # would flood the very log panel that's polling.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    return logging.getLogger("web")
