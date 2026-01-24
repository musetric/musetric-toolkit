from pathlib import Path

from musetric_toolkit.common import envs
from musetric_toolkit.common.model_files import ensure_model_file, ensure_model_files
from musetric_toolkit.separate_audio.bs_roformer_separator import BSRoformerSeparator
from musetric_toolkit.separate_audio.mdx_net_separator import MDXNetSeparator
from musetric_toolkit.separate_audio.system_info import (
    ensure_ffmpeg,
    print_acceleration_info,
    setup_torch_optimization,
)


def main(args) -> None:
    models_root = Path(args.models_path)
    model_checkpoint_path = models_root / envs.model_checkpoint_rel_path
    model_config_path = models_root / envs.model_config_rel_path
    karaoke_mdx_model_path = models_root / envs.karaoke_mdx_model_rel_path
    mdx_model_data_path = models_root / envs.mdx_model_data_rel_path

    ensure_model_files(
        model_checkpoint_path,
        model_config_path,
    )
    ensure_model_file(
        envs.karaoke_mdx_model_url,
        karaoke_mdx_model_path,
        "MDX karaoke model",
    )
    ensure_model_file(
        envs.mdx_model_data_url,
        mdx_model_data_path,
        "MDX model data",
    )
    ensure_ffmpeg()
    print_acceleration_info()
    setup_torch_optimization()

    separator = BSRoformerSeparator(
        model_checkpoint_path=model_checkpoint_path,
        model_config_path=model_config_path,
        sample_rate=args.sample_rate,
    )
    separator.separate_audio(
        source_path=args.source_path,
        vocal_path=args.lead_path,
        instrumental_path=args.instrumental_path,
    )

    lead_back_separator = MDXNetSeparator(
        model_path=karaoke_mdx_model_path,
        model_data_path=mdx_model_data_path,
        sample_rate=args.sample_rate,
    )
    lead_back_separator.separate_audio(
        source_path=args.lead_path,
        vocal_path=args.lead_path,
        instrumental_path=args.backing_path,
    )
