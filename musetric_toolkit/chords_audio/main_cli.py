import argparse
import logging
import os
import sys
import warnings

from musetric_toolkit.common.logger import redirect_std_streams, setup_logging
from musetric_toolkit.common.paths import default_models_path


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
        message=r".*n_fft=.* is too large for input signal.*",
        category=UserWarning,
    )


def _tune_logger(logger: logging.Logger, target_level: int) -> None:
    logger.setLevel(target_level)
    logger.propagate = True
    if logger.handlers:
        logger.handlers.clear()


def configure_third_party_logging(log_level: str) -> None:
    if log_level == "debug":
        return
    target_level = logging.ERROR if log_level == "error" else logging.WARNING
    for logger_name in ("torch", "torchaudio", "librosa", "numba"):
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
        description="Detect chords with timings using ChordMini",
    )
    parser.add_argument(
        "--audio-path",
        required=True,
        help="Path to audio file",
    )
    parser.add_argument(
        "--result-path",
        required=True,
        help="Path to write chords JSON result",
    )
    parser.add_argument(
        "--models-path",
        default=default_models_path(),
        help=(
            "Set HF_HOME, HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE,"
            " TORCH_HOME and the ChordMini checkpoint cache to this path"
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
    redirect_std_streams()

    try:
        from musetric_toolkit.chords_audio.main import main  # noqa: PLC0415

        main(args)
    except Exception:
        logging.exception("Chord detection failed")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
