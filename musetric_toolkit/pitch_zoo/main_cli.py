import argparse
import importlib
import logging
import sys

from musetric_toolkit.common.logger import redirect_std_streams, setup_logging
from musetric_toolkit.common.paths import default_models_path
from musetric_toolkit.pitch_audio.main_cli import (
    apply_models_path,
    configure_warning_filters,
)
from musetric_toolkit.pitch_zoo.registry import ZOO_MODELS

TRUTH_SET_NAMES = ("vocadito", "dcs", "ptdb")
PLOT_SELECTIONS = ("worst", "disputes", "differ")
COMMAND_MODULES = {
    "run": "musetric_toolkit.pitch_zoo.run",
    "align": "musetric_toolkit.pitch_zoo.align",
    "truth": "musetric_toolkit.pitch_zoo.truth",
    "resynth": "musetric_toolkit.pitch_zoo.resynth",
    "lag": "musetric_toolkit.pitch_zoo.lag",
    "score": "musetric_toolkit.pitch_zoo.score",
    "parity": "musetric_toolkit.pitch_zoo.parity",
    "disputes": "musetric_toolkit.pitch_zoo.disputes",
    "plot": "musetric_toolkit.pitch_zoo.plot",
}


def _tracks_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracks-dir",
        required=True,
        help="Directory of <track>/<name>.csv folders",
    )


def _hop_ms(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=5.0,
        help="Frame step of the grid in milliseconds",
    )


def _add_run(commands, common: argparse.ArgumentParser) -> None:
    run = commands.add_parser(
        "run",
        parents=[common],
        help="run pitch models on audio and write f0 per frame on the bench grid",
    )
    run.add_argument(
        "--audio-path",
        required=True,
        help="Audio file, or a directory whose audio files are all processed",
    )
    run.add_argument(
        "--out-dir",
        required=True,
        help="Directory for <stem>/<model>.csv (time_s,f0_hz,confidence)",
    )
    run.add_argument(
        "--models",
        nargs="+",
        default=list(ZOO_MODELS),
        choices=list(ZOO_MODELS),
        help="Models to run (default: all)",
    )
    _hop_ms(run)


def _add_align(commands, common: argparse.ArgumentParser) -> None:
    align = commands.add_parser(
        "align",
        parents=[common],
        help="check the timing of models on synthetic vibrato and glides",
    )
    align.add_argument(
        "--models",
        nargs="+",
        default=list(ZOO_MODELS),
        choices=list(ZOO_MODELS),
        help="Models to check (default: all)",
    )
    align.add_argument("--out", default=None, help="Markdown file for the table")


def _add_truth(commands, common: argparse.ArgumentParser) -> None:
    truth = commands.add_parser(
        "truth",
        parents=[common],
        help="fetch openly licensed sets with ground-truth f0 and convert them",
    )
    truth.add_argument(
        "--data-dir",
        required=True,
        help="Directory for <set>/audio/<track>.wav and "
        "<set>/tracks/<track>/truth.csv",
    )
    truth.add_argument(
        "--sets",
        nargs="+",
        default=list(TRUTH_SET_NAMES),
        choices=TRUTH_SET_NAMES,
        help="Sets to fetch (default: all)",
    )
    _hop_ms(truth)


def _add_resynth(commands, common: argparse.ArgumentParser) -> None:
    resynth = commands.add_parser(
        "resynth",
        parents=[common],
        help="resynthesize vocals with WORLD along a known f0 curve",
    )
    resynth.add_argument(
        "--audio-path",
        required=True,
        help="Vocal file, or a directory whose audio files are all processed",
    )
    resynth.add_argument(
        "--f0-dir",
        required=True,
        help="Directory of <stem>/<f0-name>.csv curves to synthesize along",
    )
    resynth.add_argument(
        "--f0-name",
        default="reference",
        help="Name of the curve CSV (default: reference)",
    )
    resynth.add_argument(
        "--data-dir",
        required=True,
        help="Directory for <name>/audio/<stem>.flac and "
        "<name>/tracks/<stem>/truth.csv",
    )
    resynth.add_argument(
        "--name",
        default="resynth",
        help="Name of the written set (default: resynth)",
    )
    _hop_ms(resynth)


def _add_lag(commands, common: argparse.ArgumentParser) -> None:
    lag = commands.add_parser(
        "lag",
        parents=[common],
        help="find the time shift of a truth track against aligned models",
    )
    _tracks_dir(lag)
    lag.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Names of the model CSVs to measure the shift with",
    )
    lag.add_argument(
        "--truth",
        default="truth",
        help="Name of the truth CSV (default: truth)",
    )
    lag.add_argument(
        "--range-ms",
        type=float,
        default=40.0,
        help="Largest shift tried in each direction, in milliseconds",
    )


def _add_score(commands, common: argparse.ArgumentParser) -> None:
    score = commands.add_parser(
        "score",
        parents=[common],
        help="score pitch tracks against a truth track with the bench metrics",
    )
    _tracks_dir(score)
    score.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Names of the CSVs to score, such as rmvpe fcpe reference pr976",
    )
    score.add_argument(
        "--truth",
        default="truth",
        help="Name of the CSV scored against (default: truth)",
    )
    score.add_argument(
        "--no-voicing",
        dest="voicing",
        action="store_false",
        help="Leave out the voicing columns, for a truth whose unvoiced frames "
        "are not reliable",
    )
    score.add_argument(
        "--worst-count",
        type=int,
        default=5,
        help="Worst 5 s windows kept per track in the JSON",
    )
    score.add_argument(
        "--out",
        default=None,
        help="Path prefix of the .md and .json report (default: "
        "<tracks-dir>/score-<truth>)",
    )


def _add_parity(commands, common: argparse.ArgumentParser) -> None:
    parity = commands.add_parser(
        "parity",
        parents=[common],
        help="check the scores against a corpus report of the Musetric bench",
    )
    parity.add_argument(
        "--tracks-dir",
        required=True,
        help="Directory the bench wrote its track folders to",
    )
    parity.add_argument(
        "--bench-report",
        required=True,
        help="<tag>.corpus.json written by measure:pitch compare",
    )
    parity.add_argument("--tag", required=True, help="Tag of the scored CSVs")


def _add_disputes(commands, common: argparse.ArgumentParser) -> None:
    disputes = commands.add_parser(
        "disputes",
        parents=[common],
        help="list voiced but untrusted runs of a reference with every model",
    )
    _tracks_dir(disputes)
    disputes.add_argument(
        "--name",
        default="reference",
        help="Name of the reference CSV, as musetric-pitch writes it "
        "(default: reference)",
    )
    disputes.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model CSVs to show next to the reference",
    )
    disputes.add_argument(
        "--min-ms",
        type=float,
        default=50.0,
        help="Shortest run listed, in milliseconds",
    )
    disputes.add_argument(
        "--count",
        type=int,
        default=50,
        help="Longest runs listed",
    )


def _add_plot(commands, common: argparse.ArgumentParser) -> None:
    plot = commands.add_parser(
        "plot",
        parents=[common],
        help="draw spectrum.png and overlay.png for windows of two pitch tracks",
    )
    _tracks_dir(plot)
    plot.add_argument(
        "--audio-dir",
        required=True,
        help="Directory with <track>.<ext> audio for the spectrum",
    )
    plot.add_argument(
        "--reference",
        default="reference",
        help="CSV drawn on the spectrum and in the lower panel " "(default: reference)",
    )
    plot.add_argument(
        "--compare",
        required=True,
        help="CSV drawn in the upper panel, such as a model or a tracker",
    )
    plot.add_argument(
        "--select",
        default="worst",
        choices=PLOT_SELECTIONS,
        help="worst: the worst 5 s windows of --compare against --reference; "
        "disputes: the longest voiced but untrusted runs of --reference; "
        "differ: where frames trusted by --compare are over 50 cents from "
        "--reference",
    )
    plot.add_argument("--count", type=int, default=10, help="Windows drawn")
    plot.add_argument(
        "--window",
        action="append",
        default=None,
        help="Draw <track>:<from>-<to> instead of selecting; repeatable",
    )
    plot.add_argument("--out-dir", required=True, help="Directory for the windows")
    plot.add_argument(
        "--dpi", type=int, default=150, help="Resolution of the images (default: 150)"
    )
    plot.add_argument("--reference-title", default=None, help="Lower panel title")
    plot.add_argument("--compare-title", default=None, help="Upper panel title")


def parse_arguments():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--models-path",
        default=default_models_path(),
        help="Directory for downloaded checkpoints",
    )
    common.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warn", "error"],
        help="Set the logging level",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Check the pitch reference of the Musetric bench: run pitch models "
            "side by side, measure them, the reference and the trackers on ground "
            "truth, and look at the result"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for add in (
        _add_run,
        _add_align,
        _add_truth,
        _add_resynth,
        _add_lag,
        _add_score,
        _add_parity,
        _add_disputes,
        _add_plot,
    ):
        add(commands, common)
    return parser.parse_args()


def main_cli() -> None:
    args = parse_arguments()
    apply_models_path(args.models_path)
    configure_warning_filters(args.log_level)
    setup_logging(args.log_level)
    redirect_std_streams()

    try:
        importlib.import_module(COMMAND_MODULES[args.command]).main(args)
    except Exception:
        logging.exception("Pitch zoo %s failed", args.command)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
