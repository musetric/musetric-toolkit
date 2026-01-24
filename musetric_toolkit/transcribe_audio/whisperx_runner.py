import collections
import logging
import re
import shutil
import sys
import typing
import warnings
from contextlib import contextmanager, suppress
from importlib.util import find_spec
from pathlib import Path

import omegaconf.base
import omegaconf.dictconfig
import omegaconf.listconfig
import omegaconf.nodes
import pyannote.audio.core.model
import torch
import whisperx
from lightning.pytorch import __version__ as pl_version
from lightning.pytorch.utilities.migration import migrate_checkpoint, pl_legacy_patch
from packaging.version import Version

from musetric_toolkit.common.logger import send_message
from musetric_toolkit.separate_audio.system_info import (
    print_acceleration_info,
    setup_torch_optimization,
)
from musetric_toolkit.transcribe_audio.download_progress import (
    intercept_hf_downloads,
)
from musetric_toolkit.transcribe_audio.language_detector import (
    detect_language_fulltrack,
)
from musetric_toolkit.transcribe_audio.silence_filter import (
    filter_silent_segments,
)

_AUDIO_SHORT_WARNING = re.compile(r"Audio is shorter than 30s", re.IGNORECASE)


class SuppressMessageFilter(logging.Filter):
    def __init__(self, patterns: list[re.Pattern[str]]) -> None:
        super().__init__()
        self._patterns = patterns

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(pattern.search(message) for pattern in self._patterns)


def _tune_logger(logger: logging.Logger, target_level: int) -> None:
    logger.setLevel(target_level)
    logger.propagate = True
    if logger.handlers:
        logger.handlers.clear()


def _matches_prefix(logger_name: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        logger_name == prefix or logger_name.startswith(f"{prefix}.")
        for prefix in prefixes
    )


def configure_warning_filters(log_level: str) -> None:
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    if log_level == "debug":
        return
    warnings.filterwarnings(
        "ignore",
        message=r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Module 'speechbrain\.pretrained' was deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
        category=UserWarning,
    )


def configure_third_party_logging(log_level: str) -> None:
    if log_level == "debug":
        return
    target_level = logging.ERROR if log_level == "error" else logging.WARNING
    prefixes = (
        "huggingface_hub",
        "lightning",
        "pytorch_lightning",
        "pyannote",
        "pyannote.audio",
        "speechbrain",
        "torchaudio",
        "whisperx",
    )

    for logger_name in prefixes:
        _tune_logger(logging.getLogger(logger_name), target_level)

    for logger_name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and _matches_prefix(
            logger_name, prefixes
        ):
            _tune_logger(logger, target_level)

    suppress_filter = SuppressMessageFilter([_AUDIO_SHORT_WARNING])
    for logger_name in ("whisperx", "whisperx.asr"):
        logger = logging.getLogger(logger_name)
        has_filter = any(
            isinstance(existing, SuppressMessageFilter) for existing in logger.filters
        )
        if not has_filter:
            logger.addFilter(suppress_filter)


def configure_torch_serialization() -> None:
    # Allowlist safe types used by OmegaConf/typing in PyTorch weights-only loads.
    torch.serialization.add_safe_globals(
        [
            omegaconf.listconfig.ListConfig,
            omegaconf.dictconfig.DictConfig,
            omegaconf.base.ContainerMetadata,
            omegaconf.base.Metadata,
            omegaconf.nodes.AnyNode,
            typing.Any,
            int,
            list,
            dict,
            tuple,
            set,
            collections.defaultdict,
            collections.OrderedDict,
            torch.torch_version.TorchVersion,
            pyannote.audio.core.model.Introspection,
        ]
    )


@contextmanager
def allow_unsafe_torch_load():
    original_torch_load = torch.load
    original_serialization_load = torch.serialization.load

    def torch_load_unrestricted(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_unrestricted
    torch.serialization.load = torch_load_unrestricted
    try:
        yield
    finally:
        torch.load = original_torch_load
        torch.serialization.load = original_serialization_load


def maybe_upgrade_whisperx_checkpoint() -> None:
    try:
        spec = find_spec("whisperx")
        if not spec or not spec.origin:
            return
        checkpoint_path = (
            Path(spec.origin).resolve().parent / "assets" / "pytorch_model.bin"
        )
        if not checkpoint_path.is_file():
            return

        with pl_legacy_patch():
            checkpoint = torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
                weights_only=False,
            )

        ckpt_version = checkpoint.get("pytorch-lightning_version")
        if not ckpt_version:
            return
        if Version(ckpt_version) >= Version(pl_version):
            return

        backup_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.bak")
        if not backup_path.exists():
            shutil.copyfile(checkpoint_path, backup_path)

        migrate_checkpoint(checkpoint)
        torch.save(checkpoint, checkpoint_path)
    except Exception as error:
        logging.debug("WhisperX checkpoint upgrade skipped: %s", error)


_PROGRESS_PATTERN = re.compile(r"^Progress:\s+([0-9]+(?:\.[0-9]+)?)%")


class ProgressTracker:
    def __init__(self, min_delta: float = 0.01) -> None:
        self._min_delta = min_delta
        self._last = -1.0

    def report_fraction(self, progress: float) -> None:
        progress = max(0.0, min(progress, 1.0))
        if progress <= self._last:
            return
        if (
            self._last >= 0.0
            and progress < self._last + self._min_delta
            and progress < 1.0
        ):
            return
        self._last = progress
        send_message({"type": "progress", "progress": progress})

    def report_percent(self, percent: float) -> None:
        self.report_fraction(percent / 100.0)

    def ensure_minimum(self, progress: float) -> None:
        if progress > self._last:
            self.report_fraction(progress)

    def finalize(self) -> None:
        self.report_fraction(1.0)


class ProgressLineInterceptor:
    def __init__(self, stream, tracker: ProgressTracker) -> None:
        self._stream = stream
        self._tracker = tracker
        self._buffer = ""
        self.encoding = getattr(stream, "encoding", "utf-8")
        self.errors = getattr(stream, "errors", "replace")

    def _maybe_report_progress(self, line: str) -> bool:
        match = _PROGRESS_PATTERN.match(line)
        if not match:
            return False
        try:
            percent = float(match.group(1))
        except ValueError:
            return False
        self._tracker.report_percent(percent)
        return True

    def write(self, message: str | bytes) -> int:
        if not message:
            return 0
        if isinstance(message, bytes):
            message = message.decode(self.encoding, errors=self.errors)
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue
            if self._maybe_report_progress(line):
                continue
            self._stream.write(line + "\n")
        return len(message)

    def flush(self) -> None:
        if self._buffer:
            line = self._buffer.rstrip("\r")
            if line and not self._maybe_report_progress(line):
                self._stream.write(line)
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


@contextmanager
def intercept_progress_lines(tracker: ProgressTracker):
    original_stdout = sys.stdout
    wrapper = ProgressLineInterceptor(original_stdout, tracker)
    sys.stdout = wrapper
    try:
        yield
    finally:
        sys.stdout = original_stdout
        wrapper.flush()


def transcribe_with_whisperx(audio_path: str, log_level: str = "info"):
    configure_warning_filters(log_level)
    configure_third_party_logging(log_level)
    configure_torch_serialization()
    maybe_upgrade_whisperx_checkpoint()
    print_acceleration_info()
    setup_torch_optimization()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    with allow_unsafe_torch_load(), intercept_hf_downloads("WhisperX model"):
        model = whisperx.load_model(
            "large-v3",
            device,
            compute_type=compute_type,
            vad_method="pyannote",
            vad_options={
                "vad_onset": 0.65,
                "vad_offset": 0.50,
            },
        )
    audio = whisperx.load_audio(audio_path)
    detected_language = detect_language_fulltrack(
        model,
        audio,
        sample_rate=16000,
    )
    progress_tracker = ProgressTracker()
    progress_tracker.report_fraction(0.0)
    with intercept_progress_lines(progress_tracker):
        result = model.transcribe(
            audio,
            batch_size=1,
            language=detected_language,
            chunk_size=10,
            print_progress=True,
            combined_progress=True,
        )
        progress_tracker.ensure_minimum(0.5)

    segments = filter_silent_segments(result.get("segments", []), audio)
    detected_language = result.get("language", detected_language)

    try:
        with (
            allow_unsafe_torch_load(),
            intercept_hf_downloads("WhisperX alignment model"),
        ):
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_language,
                device=device,
            )
        with intercept_progress_lines(progress_tracker):
            aligned = whisperx.align(
                segments,
                align_model,
                metadata,
                audio,
                device,
                return_char_alignments=False,
                print_progress=True,
                combined_progress=True,
            )
        segments = aligned.get("segments", segments)
        detected_language = aligned.get("language", detected_language)
    except Exception as align_error:
        logging.warning("Alignment skipped: %s", align_error)
    finally:
        progress_tracker.finalize()

    return segments, detected_language
