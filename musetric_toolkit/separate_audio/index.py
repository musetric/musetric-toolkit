import argparse
import logging
import sys

from musetric_toolkit.common import envs
from musetric_toolkit.common.logger import redirect_std_streams, setup_logging
from musetric_toolkit.common.model_files import ensure_model_file, ensure_model_files
from musetric_toolkit.separate_audio.bs_roformer_separator import BSRoformerSeparator
from musetric_toolkit.separate_audio.mdx_net_separator import MDXNetSeparator
from musetric_toolkit.separate_audio.system_info import (
    ensure_ffmpeg,
    print_acceleration_info,
    setup_torch_optimization,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Separate audio into lead vocals, backing vocals, and instrumental parts"
        )
    )
    parser.add_argument(
        "--source-path", required=True, help="Path to source audio file"
    )
    parser.add_argument(
        "--lead-path", required=True, help="Path for lead vocal output file"
    )
    parser.add_argument(
        "--backing-path", required=True, help="Path for backing vocal output file"
    )
    parser.add_argument(
        "--instrumental-path",
        required=True,
        help="Path for instrumental output file",
    )
    parser.add_argument(
        "--sample-rate",
        required=True,
        type=int,
        help="Sample rate for separation",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warn", "error"],
        help="Set the logging level",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    setup_logging(args.log_level)
    redirect_std_streams()

    try:
        ensure_model_files(
            envs.model_checkpoint_path,
            envs.model_config_path,
        )
        ensure_model_file(
            envs.karaoke_mdx_model_url,
            envs.karaoke_mdx_model_path,
            "MDX karaoke model",
        )
        ensure_model_file(
            envs.mdx_model_data_url,
            envs.mdx_model_data_path,
            "MDX model data",
        )
        ensure_ffmpeg()
        print_acceleration_info()
        setup_torch_optimization()

        separator = BSRoformerSeparator(
            model_checkpoint_path=envs.model_checkpoint_path,
            model_config_path=envs.model_config_path,
            sample_rate=args.sample_rate,
        )
        separator.separate_audio(
            source_path=args.source_path,
            vocal_path=args.lead_path,
            instrumental_path=args.instrumental_path,
        )

        lead_back_separator = MDXNetSeparator(
            model_path=envs.karaoke_mdx_model_path,
            model_data_path=envs.mdx_model_data_path,
            sample_rate=args.sample_rate,
        )
        lead_back_separator.separate_audio(
            source_path=args.lead_path,
            vocal_path=args.lead_path,
            instrumental_path=args.backing_path,
        )
    except Exception as error:
        logging.error("Audio separation failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
