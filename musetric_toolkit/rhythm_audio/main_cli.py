import argparse
import logging
import os
import re
import sys
import warnings

from musetric_toolkit.common.logger import redirect_std_streams, setup_logging
from musetric_toolkit.common.paths import default_models_path

_TORCH_HUB_PROGRESS = re.compile(r"^\r?\d+(?:\.\d+)?%$")


def build_stream_suppression_patterns(
    log_level: str,
) -> list[re.Pattern[str]]:
    if log_level == "debug":
        return []
    return [_TORCH_HUB_PROGRESS]


def _tune_logger(logger: logging.Logger, target_level: int) -> None:
    logger.setLevel(target_level)
    logger.propagate = True
    if logger.handlers:
        logger.handlers.clear()


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
        message=r".*load_with_torchcodec.*",
        category=UserWarning,
    )


def configure_third_party_logging(log_level: str) -> None:
    if log_level == "debug":
        return
    target_level = logging.ERROR if log_level == "error" else logging.WARNING
    for logger_name in ("torch", "torchaudio"):
        _tune_logger(logging.getLogger(logger_name), target_level)


def setup_cli_logging(log_level: str) -> None:
    configure_warning_filters(log_level)
    setup_logging(log_level)
    configure_third_party_logging(log_level)


def apply_models_path(models_path: str) -> None:
    os.environ["HF_HOME"] = models_path
    os.environ["HF_HUB_CACHE"] = models_path
    os.environ["HUGGINGFACE_HUB_CACHE"] = models_path
    os.environ["TORCH_HOME"] = models_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze rhythm (BPM, beats, downbeats) with beat_this",
    )
    parser.add_argument(
        "--audio-path",
        required=True,
        help="Path to audio file",
    )
    parser.add_argument(
        "--result-path",
        required=True,
        help="Path to write rhythm JSON result",
    )
    parser.add_argument(
        "--models-path",
        default=default_models_path(),
        help=(
            "Set HF_HOME, HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE,"
            " TORCH_HOME to this path"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warn", "error"],
        help="Set the logging level",
    )
    return parser.parse_args()


def main_cli() -> None:
    args = parse_arguments()
    apply_models_path(args.models_path)
    setup_cli_logging(args.log_level)
    redirect_std_streams(
        suppress_patterns=build_stream_suppression_patterns(args.log_level),
    )

    try:
        from musetric_toolkit.rhythm_audio.main import main  # noqa: PLC0415

        main(args)
    except Exception:
        logging.exception("Rhythm analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
