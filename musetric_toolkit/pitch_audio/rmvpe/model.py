# RMVPE network (arXiv:2306.15412) as implemented in the RVC project.
# Vendored from https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
# (MIT, see LICENSE next to this file); see thirdPartyNotices.md.
import librosa.filters
import torch
import torch.nn as nn

N_MELS = 128
N_BINS = 360


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x.contiguous())[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for c in self.conv:
            x = c(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(
                ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum)
            )
            in_channels = out_channels
            out_channels *= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat.append(t)
        return x, concat


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for _ in range(n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, (3, 3), stride, (1, 1),
                output_padding=out_padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for c2 in self.conv2:
            x = c2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat)
        return x


class E2E(nn.Module):
    def __init__(
        self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1,
        en_out_channels=16,
    ):
        super().__init__()
        self.unet = DeepUnet(
            kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_BINS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_BINS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)


class MelSpectrogram(nn.Module):
    def __init__(
        self, n_mel_channels, sampling_rate, win_length, hop_length, n_fft=None,
        mel_fmin=0, mel_fmax=None, clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels,
            fmin=mel_fmin, fmax=mel_fmax, htk=True,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("hann", torch.hann_window(win_length))
        self.clamp = clamp

    def forward(self, audio):
        fft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.hann, center=True, return_complex=True,
        )
        mag = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        mel = torch.matmul(self.mel_basis, mag)
        return torch.log(torch.clamp(mel, min=self.clamp))
