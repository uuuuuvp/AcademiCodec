# 移除归一化

from typing import Optional, Any, Dict, Tuple, List, Union
import typing as tp
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from typeguard import check_argument_types
# from funcodec.train.abs_gan_espnet_model import AbsGANESPnetModel
# from funcodec.torch_utils.device_funcs import force_gatherable
from librosa.filters import mel as librosa_mel_fn


class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=16000,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
        device='cpu'
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length, device=device).float()
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audioin, return_power_spec=False):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audioin, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        power_spec = torch.sum(torch.pow(fft, 2), dim=[-1])
        mel_output = torch.matmul(self.mel_basis, power_spec)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        if return_power_spec:
            log_power_spec = torch.log10(torch.clamp(power_spec, min=1e-5))
            return log_mel_spec, log_power_spec
        return log_mel_spec


EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]


def _linear_overlap_add(frames: tp.List[torch.Tensor], stride: int):
    assert len(frames)
    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    frame_length = frames[0].shape[-1]
    t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1: -1]
    weight = 0.5 - (t - 0.5).abs()

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0

    for frame in frames:
        frame_length = frame.shape[-1]
        out[..., offset:offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset:offset + frame_length] += weight[:frame_length]
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


class fbank():
    def __init__(
            self,
            input_size: int,
            odim: int = 512,
            encoder: torch.nn.Module = None,
            decoder: torch.nn.Module = None,
            target_sample_hz: int = 16_000,
            segment_dur: Optional[float] = 1.0,
            overlap_ratio: Optional[float] = 0.01,
            multi_spectral_window_powers_of_two: Union[Tuple, List] = tuple(range(5, 11)),
            multi_spectral_n_mels: int = 64,
            codec_domain: List = ["stft", "stft"],
            domain_conf: Optional[Dict] = {}):
        assert check_argument_types()
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sample_rate = target_sample_hz
        self.segment_dur = segment_dur
        self.overlap_ratio = overlap_ratio
        self.codec_domain = codec_domain
        self.domain_conf = domain_conf
        if codec_domain[0] in ["stft", "mag_phase", "mag_angle", "mag_oracle_phase"]:
            self.enc_trans_func = torchaudio.transforms.Spectrogram(
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
                power=None,
            )
        elif codec_domain[0] in ["mag"]:
            self.enc_trans_func = torchaudio.transforms.Spectrogram(
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
                power=1,
            )
        elif codec_domain[0] == "mel":
            self.enc_trans_func = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sample_hz,
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
                n_mels=80,
                power=2,
            )
        if codec_domain[1] in ["stft", "mag_phase", "mag_angle", "mag_oracle_phase"]:
            self.dec_trans_func = torchaudio.transforms.InverseSpectrogram(
                n_fft=domain_conf.get("n_fft", 512),
                hop_length=domain_conf.get("hop_length", 160),
            )

        # multi spectral reconstruction
        self.mel_spec_transforms = nn.ModuleList([])

        for powers in multi_spectral_window_powers_of_two:
            win_length = 2 ** powers

            melspec_transform = Audio2Mel(
                sampling_rate=target_sample_hz,
                win_length=win_length,
                hop_length=win_length // 4,
                n_mel_channels=multi_spectral_n_mels
            )

            self.mel_spec_transforms.append(melspec_transform)


    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment_dur is None:
            return None
        return int(self.segment_dur * self.sample_rate)

    @property
    def segment_stride(self) -> tp.Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap_ratio) * segment_length))

    def t2f(self, x: torch.Tensor):
        pass

    def _encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        assert x.dim() == 3
        _, channels, length = x.shape
        assert 0 < channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        # print("length:", length, "stride:", stride)
        for offset in range(0, length, stride):
            # print("start:", offset, "end:", offset + segment_length)
            frame = x[:, :, offset: offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        length = x.shape[-1]
        duration = length / self.sample_rate
        assert self.segment_dur is None or duration <= 1e-5 + self.segment_dur
        # scale = None # 因为取消归一化了

        if self.codec_domain[0] == "stft":
            x_complex = self.enc_trans_func(x.squeeze(1))
            if self.encoder.input_size == 2:
                x = torch.stack([x_complex.real, x_complex.imag], dim=1)
            else:
                x = torch.cat([x_complex.real, x_complex.imag], dim=1)
        elif self.codec_domain[0] == "mag":
            x_mag = self.enc_trans_func(x.squeeze(1))
            if self.encoder.input_size == 1:
                x = x_mag.unsqueeze(1)
            else:
                x = x_mag
        elif self.codec_domain[0] == "mag_angle":
            x_complex = self.enc_trans_func(x.squeeze(1))
            x_mag = torch.abs(x_complex)
            x_log_mag = torch.log(torch.clamp(x_mag, min=1e-6))
            x_angle = torch.angle(x_complex)
            if self.encoder.input_size == 2:
                x = torch.stack([x_log_mag, x_angle], dim=1)
            else:
                x = torch.cat([x_log_mag, x_angle], dim=1)
        elif self.codec_domain[0] == "mag_phase":
            x_complex = self.enc_trans_func(x.squeeze(1))
            x_mag = torch.abs(x_complex)
            x_log_mag = torch.log(torch.clamp(x_mag, min=1e-6))
            x_phase = x_complex / torch.clamp(x_mag, min=1e-6)
            if self.encoder.input_size == 3:
                x = torch.stack([x_log_mag, x_phase.real, x_phase.imag], dim=1)
            else:
                x = torch.cat([x_log_mag, x_phase.real, x_phase.imag], dim=1)
        elif self.codec_domain[0] == "mel":
            x = self.enc_trans_func(x.squeeze(1))
            if self.encoder.input_size == 1:
                x = x.unsqueeze(1)
        return x

    def _decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = []
        for frame in encoded_frames:
            frames.append(self._decode_frame(frame))

        return _linear_overlap_add(frames, self.segment_stride or 1)

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        codes, scale = encoded_frame
        emb = codes
        out = self.decoder(emb)
        if self.codec_domain[1] == "stft":
            if len(out.shape) == 3:
                out_list = torch.split(out, out.shape[1]//2, dim=1)
            else:
                out_list = torch.split(out, 1, dim=1)
            out = torch.complex(out_list[0], out_list[1])
            out = self.dec_trans_func(out).unsqueeze(1)
        elif self.codec_domain[1] == "mag_phase":
            if len(out.shape) == 3:
                out_list = torch.split(out, out.shape[1] // 3, dim=1)
            else:
                out_list = [x.squeeze(1) for x in torch.split(out, 1, dim=1)]
            x_mag = F.softplus(out_list[0])
            x_phase = torch.complex(out_list[1], out_list[2])
            out = x_mag * x_phase
            out = self.dec_trans_func(out).unsqueeze(1)
        elif self.codec_domain[1] == "mag_angle":
            if len(out.shape) == 3:
                out_list = torch.split(out, out.shape[1] // 2, dim=1)
            else:
                out_list = [x.squeeze(1) for x in torch.split(out, 1, dim=1)]
            x_mag = F.softplus(out_list[0])
            x_angle = torch.sin(out_list[1]) * torch.pi
            x_spec = torch.complex(torch.cos(x_angle) * x_mag, torch.sin(x_angle) * x_mag)
            out = self.dec_trans_func(x_spec).unsqueeze(1)
        elif self.codec_domain[1] == "mag_oracle_phase":
            if len(out.shape) == 4:
                out = out.squeeze(1)
            (scale, x_angle), x_mag = scale, out
            x_spec = torch.complex(torch.cos(x_angle)*x_mag, torch.sin(x_angle)*x_mag)
            out = self.dec_trans_func(x_spec).unsqueeze(1)
        elif (self.codec_domain[0] in ["stft", "mag", "mag_phase", "mag_angle", "mag_oracle_phase"] and
              self.codec_domain[1] == "time"):
            hop_length = self.domain_conf.get("hop_length", 160)
            out = out[:, :, hop_length//2: -hop_length//2]

        if scale is not None:
            out = out * scale.view(-1, 1, 1)
        return out

    def inference(
            self,
            speech: torch.Tensor,
            need_recon: bool = True,
            bit_width: int = None,
            use_scale: bool = False,
    ) -> Dict[str, torch.Tensor]:

        codes = []
        code_idxs = []
        all_sub_quants = []
        if speech.dim() == 2:
            speech = speech.unsqueeze(1)
        frames = self._encode(speech)
        for emb, scale in frames:
            bb, tt, device = emb.shape[0], emb.shape[1], emb.device
            # 因为移除量化器，所以直接使用跳过的方法
            code_embs, indices, sub_quants = emb, torch.zeros(bb, tt, dtype=torch.long, device=device), torch.zeros_like(emb, device=device)
            codes.append((code_embs, scale if use_scale else None))
            code_idxs.append(indices)
            all_sub_quants.append(sub_quants)
        recon_speech = None
        if need_recon:
            recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
        retval = dict(
            recon_speech=recon_speech,
            code_indices=code_idxs,
            code_embeddings=codes,
            sub_quants=all_sub_quants
        )
        return retval

    def inference_encoding(
            self,
            speech: torch.Tensor,
            need_recon: bool = False,
            bit_width: int = None,
            use_scale: bool = False,
    ) -> Dict[str, torch.Tensor]:

        codes = []
        code_idxs = []
        all_sub_quants = []
        if speech.dim() == 2:
            speech = speech.unsqueeze(1)
        frames = self._encode(speech)
        for emb, scale in frames:
            quant_in = emb
            # quant_out, indices, sub_quants = self.quantizer.inference(quant_in, bandwidth=bit_width)
            quant_out, indices, sub_quants = self.quantizer(quant_in, bandwidth=bit_width)
            code_embs = quant_out
            codes.append((code_embs, scale if use_scale else None))
            code_idxs.append(indices)
            all_sub_quants.append(sub_quants)
        recon_speech = None
        if need_recon:
            recon_speech = self._decode(codes)[:, :, :speech.shape[-1]]
        retval = dict(
            recon_speech=recon_speech,
            code_indices=code_idxs,
            code_embeddings=codes,
            sub_quants=all_sub_quants
        )
        return retval

    def inference_decoding(
            self,
            token_idx: torch.Tensor,
            need_recon: bool = True,
            bit_width: int = None,
            use_scale: bool = False,
    ) -> Dict[str, torch.Tensor]:

        codes = []
        token_idx = token_idx.permute(2, 0, 1).unsqueeze(0)
        for tokens in token_idx:
            code_embs = self.quantizer.decode(tokens)
            codes.append((code_embs.transpose(1, 2), None))
        recon_speech = None
        if need_recon:
            recon_speech = self._decode(codes)
        retval = dict(
            recon_speech=recon_speech,
            code_indices=None,
            code_embeddings=codes,
            sub_quants=None
        )
        return retval

    def inference_decoding_emb(
            self,
            token_idx: torch.Tensor,
            need_recon: bool = True,
            bit_width: int = None,
            use_scale: bool = False,
    ) -> Dict[str, torch.Tensor]:
        
        codes = [(token_idx, None)]
        recon_speech = None
        if need_recon:
            recon_speech = self._decode(codes)
        retval = dict(
            recon_speech=recon_speech,
            code_indices=None,
            code_embeddings=codes,
            sub_quants=None
        )
        return retval
