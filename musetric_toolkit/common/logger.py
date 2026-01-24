import json
import logging
import re
import sys
from contextlib import suppress
from typing import Any, ClassVar


class JSONFormatter(logging.Formatter):
    level_map: ClassVar[dict[str, str]] = {
        "DEBUG": "debug",
        "INFO": "info",
        "WARNING": "warn",
        "ERROR": "error",
    }

    def format(self, record):
        mapped_level = self.level_map.get(record.levelname, "info")
        log_entry = {"level": mapped_level, "message": record.getMessage()}
        if record.exc_info:
            log_entry["traceback"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class StreamToLogger:
    def __init__(
        self,
        logger: logging.Logger,
        level: int,
        stream,
        suppress_patterns: list[re.Pattern[str]] | None = None,
    ) -> None:
        self._logger = logger
        self._level = level
        self._stream = stream
        self._buffer = ""
        self._suppress_patterns = suppress_patterns or []
        self.encoding = getattr(stream, "encoding", "utf-8")
        self.errors = getattr(stream, "errors", "replace")

    def _should_suppress(self, line: str) -> bool:
        return any(pattern.search(line) for pattern in self._suppress_patterns)

    def write(self, message: str | bytes) -> int:
        if not message:
            return 0
        if isinstance(message, bytes):
            message = message.decode(self.encoding, errors=self.errors)
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line and not self._should_suppress(line):
                self._logger.log(self._level, line)
        return len(message)

    def flush(self) -> None:
        if self._buffer:
            line = self._buffer.rstrip("\r")
            if line and not self._should_suppress(line):
                self._logger.log(self._level, line)
            self._buffer = ""
        with suppress(Exception):
            self._stream.flush()

    def isatty(self) -> bool:
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def fileno(self) -> int:
        try:
            return self._stream.fileno()
        except Exception:
            return -1

    def writable(self) -> bool:
        return True


def setup_logging(level: str):
    level_map = {
        "debug": "DEBUG",
        "info": "INFO",
        "warn": "WARNING",
        "error": "ERROR",
    }

    handler = logging.StreamHandler(sys.__stderr__)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    python_level = level_map.get(level, "INFO")
    numeric_level = getattr(logging, python_level)
    root_logger.setLevel(numeric_level)
    handler.setLevel(numeric_level)

    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)

    root_logger.addHandler(handler)

    logging.captureWarnings(True)


def redirect_std_streams(
    stdout_level: str = "info",
    stderr_level: str = "warn",
    suppress_patterns: list[re.Pattern[str]] | None = None,
) -> None:
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "error": logging.ERROR,
    }
    if not isinstance(sys.stdout, StreamToLogger):
        sys.stdout = StreamToLogger(
            logging.getLogger("stdout"),
            level_map.get(stdout_level, logging.INFO),
            sys.__stdout__,
            suppress_patterns=suppress_patterns,
        )
    if not isinstance(sys.stderr, StreamToLogger):
        sys.stderr = StreamToLogger(
            logging.getLogger("stderr"),
            level_map.get(stderr_level, logging.WARNING),
            sys.__stderr__,
            suppress_patterns=suppress_patterns,
        )


def send_message(message: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(message) + "\n")
    sys.__stdout__.flush()
