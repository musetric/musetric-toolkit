from pathlib import Path

# Third-party model/data sources used by the audio workflows.
# See thirdPartyNotices.md for source, license, and attribution details.
model_checkpoint_url = "https://huggingface.co/SYH99999/MelBandRoformerBigSYHFTV1Fast/resolve/main/MelBandRoformerBigSYHFTV1.ckpt"
model_config_url = "https://huggingface.co/SYH99999/MelBandRoformerBigSYHFTV1Fast/resolve/main/config.yaml"

model_mel_band_roformer_dir = "mel_band_roformer_big_syhft_v1"
model_checkpoint_rel_path = Path(model_mel_band_roformer_dir) / "model.ckpt"
model_config_rel_path = Path(model_mel_band_roformer_dir) / "config.yaml"

karaoke_mdx_model_url = (
    "https://huggingface.co/AI4future/RVC/resolve/main/UVR_MDXNET_KARA_2.onnx"
)
karaoke_mdx_models_dir = "uvr_mdxnet_kara_2"
karaoke_mdx_model_rel_path = Path(karaoke_mdx_models_dir) / "UVR_MDXNET_KARA_2.onnx"

mdx_model_data_url = (
    "https://raw.githubusercontent.com/TRvlvr/application_data/main/"
    "mdx_model_data/model_data_new.json"
)
mdx_model_data_dir = "mdx_model_data"
mdx_model_data_rel_path = Path(mdx_model_data_dir) / "model_data_new.json"
