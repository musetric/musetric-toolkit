import argparse
import logging
import os
import sys
import warnings

from musetric_toolkit.common.logger import redirect_std_streams, setup_logging
from musetric_toolkit.common.paths import default_models_path


def configure_warning_filters(log_level: str) -> None:
    if log_level == "debug":
        return
    warnings.filterwarnings("ignore", category=UserWarning)


def apply_models_path(models_path: str) -> None:
    os.environ["HF_HOME"] = models_path
    os.environ["HF_HUB_CACHE"] = models_path
    os.environ["HUGGINGFACE_HUB_CACHE"] = models_path
    os.environ["TORCH_HOME"] = models_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a reference pitch track with RMVPE and a global Viterbi decode"
        ),
    )
    parser.add_argument(
        "--audio-path",
        required=True,
        help="Path to audio file",
    )
    parser.add_argument(
        "--result-path",
        required=True,
        help="Path to write the CSV (time_s,f0_hz,confidence,trusted)",
    )
    parser.add_argument(
        "--from-seconds",
        type=float,
        default=None,
        help="Start of the analysed range in seconds (default: track start)",
    )
    parser.add_argument(
        "--to-seconds",
        type=float,
        default=None,
        help="End of the analysed range in seconds (default: track end)",
    )
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=5.0,
        help="Frame step in milliseconds",
    )
    parser.add_argument(
        "--models-path",
        default=default_models_path(),
        help="Directory for the downloaded checkpoint",
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
    configure_warning_filters(args.log_level)
    setup_logging(args.log_level)
    redirect_std_streams()

    try:
        from musetric_toolkit.pitch_audio.main import main  # noqa: PLC0415

        main(args)
    except Exception:
        logging.exception("Pitch reference extraction failed")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
