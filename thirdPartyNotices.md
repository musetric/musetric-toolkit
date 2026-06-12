# Third-party Notices

This project includes or adapts portions of the following third-party works.
This notice focuses on vendored code, adapted implementations, model/data
artifacts downloaded by the toolkit, and primary runtime solutions used by the
CLI workflows. Ordinary Python package dependencies keep their license metadata
in their distributed packages.

## Ultimate Vocal Remover GUI / MDX-Net

- Source: https://github.com/Anjok07/ultimatevocalremovergui
- Usage: MDX-Net ONNX inference flow, model tensor layout, chunked demix / overlap-add behavior, and compatibility with UVR-trained MDX models.
- Local files: `musetric_toolkit/separate_audio/mdx_net_separator.py`, `musetric_toolkit/separate_audio/main.py`.
- License: MIT.
- License source: upstream README; no repository license file was found when this notice was prepared.
- Credit: Ultimate Vocal Remover GUI / UVR developers, including Anjok07 and aufr33; original MDX-Net AI code credited upstream to Kuielab and Woosung Choi.

The upstream repository README asks third-party application developers who use UVR models to credit UVR and its developers.

## AI4future/RVC - UVR_MDXNET_KARA_2.onnx

- Source: https://huggingface.co/AI4future/RVC
- Usage: `UVR_MDXNET_KARA_2.onnx` model downloaded at runtime for lead/backing vocal separation.
- Local files: `musetric_toolkit/common/envs.py`, `musetric_toolkit/separate_audio/main.py`.
- License: MIT.
- License source: Hugging Face model card metadata.

## TRvlvr/application_data

- Source: https://github.com/TRvlvr/application_data
- Usage: MDX model hash-to-parameter metadata (`mdx_model_data/model_data_new.json`) used to configure UVR-compatible MDX models.
- Local files: `musetric_toolkit/common/envs.py`, `musetric_toolkit/separate_audio/mdx_net_separator.py`.
- License: no repository license file was found when this notice was prepared.

## DeepExtract

- Source: https://github.com/abdozmantar/deepextract
- Usage: STFT / inverse-STFT tensor conversion pattern used by the MDX ONNX separator.
- Local files: `musetric_toolkit/separate_audio/mdx_net_separator.py`.
- License: MIT.
- License source: upstream `LICENSE`.

MIT License

Copyright (c) [2024] Abdullah Ozmantar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## BS-RoFormer

- Source: https://github.com/lucidrains/BS-RoFormer
- Usage: `MelBandRoformer` and attention model architecture adapted for local separation checkpoints.
- Local files: `musetric_toolkit/separate_audio/roformer/mel_band_roformer.py`, `musetric_toolkit/separate_audio/roformer/attend.py`.
- License: MIT.
- License source: upstream `LICENSE`.

MIT License

Copyright (c) 2023 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## MelBandRoformerBigSYHFTV1Fast

- Source: https://huggingface.co/SYH99999/MelBandRoformerBigSYHFTV1Fast
- Usage: checkpoint and config downloaded at runtime for vocal/instrumental separation.
- Local files: `musetric_toolkit/common/envs.py`, `musetric_toolkit/separate_audio/main.py`, `musetric_toolkit/separate_audio/mel_band_roformer_separator.py`.
- License: MIT.
- License source: Hugging Face model card metadata.

## python-audio-separator

- Source: https://github.com/nomadkaraoke/python-audio-separator
- Usage: Research tool that helped validate the BS-RoFormer approach and integration patterns.
- Local files: `musetric_toolkit/separate_audio/mel_band_roformer_separator.py`, `musetric_toolkit/separate_audio/roformer_utils.py`.
- License: MIT.
- License source: upstream `LICENSE`.

MIT License

Copyright (c) 2023 karaokenerds

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## WhisperX

- Source: https://github.com/m-bain/whisperX
- Usage: Speech-to-text and word-level alignment for `musetric-transcribe`.
- Local files: `musetric_toolkit/transcribe_audio/whisperx_runner.py`, `musetric_toolkit/transcribe_audio/language_detector.py`, `musetric_toolkit/transcribe_audio/main.py`.
- License: BSD 2-Clause.
- License source: installed package license file and upstream `LICENSE`.

BSD 2-Clause License

Copyright (c) 2024, Max Bain

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## ChordMini

- Source: https://github.com/ptnghia-j/ChordMini
- Usage: vendored inference subset under `musetric_toolkit/chords_audio/chordmini`, plus `2e1d_model_best.pth` checkpoint downloaded at runtime.
- Local files: `musetric_toolkit/chords_audio/chordmini/`, `musetric_toolkit/chords_audio/chordmini_runner.py`, `musetric_toolkit/chords_audio/chordmini_checkpoint.py`.
- License: MIT.
- License source: upstream `LICENSE` and vendored `musetric_toolkit/chords_audio/chordmini/LICENSE`.
- Local vendoring details: see `musetric_toolkit/chords_audio/chordmini/NOTICE.md`.

MIT License

Copyright (c) 2026 ChordMini contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## SKey

- Source: https://github.com/deezer/skey
- Usage: musical key detection through the `skey` Python package and its default checkpoint.
- Local files: `musetric_toolkit/key_audio/skey_runner.py`, `musetric_toolkit/key_audio/main.py`.
- License: MIT.
- License source: installed package license file and upstream `LICENSE`.

MIT License

Copyright (c) 2019-present, Deezer SA.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Beat This!

- Source: https://github.com/CPJKU/beat_this
- Usage: beat and downbeat tracking through the `beat-this` Python package and `beat_this-final0.ckpt` checkpoint.
- Local files: `musetric_toolkit/rhythm_audio/beat_this_runner.py`, `musetric_toolkit/rhythm_audio/main.py`, `musetric_toolkit/rhythm_audio/bpm_estimator.py`.
- License: MIT.
- License source: installed package license file and upstream `LICENSE`.

MIT License

Copyright (c) 2024 Institute of Computational Perception, JKU Linz, Austria

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
