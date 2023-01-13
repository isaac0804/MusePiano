import torch
import torchaudio
import numpy as np
import pydub

from utils import audio_util

class SpectrogramConverater:
    """
    Interface for converting spectrogram to and from audio using torchaudio.

    Note:
    Both Griffin Lim algorithm and Mel scaling are lossy.
    """

    def __init__(self, params, device: str = "cuda"):
        self.p = params
        self.device = torch.device(device)

        # Convert from audio to spectrogram
        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html
        self.spectrogram_func = torchaudio.transforms.Spectrogram(
            sample_rate=params.sample_rate,
            n_fft=params.n_fft,
            win_length=params.win_length,
            hop_length=params.hop_length,
            normalized=False,
            center=True,
            pad_mode="reflect"
        ).to(self.device)
        
        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.GriffinLim.html
        self.inverse_spectrogram_func = torchaudio.transforms.GriffinLim(
            n_fft=params.n_fft,
            n_iter=params.num_griffin_lim_iters,
            win_length=params.win_length,
            hop_length=params.hop_length,
            power=1.0,
        ).to(self.device)

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.MelScale.html
        self.mel_scaler = torchaudio.transforms.MelScale(
            n_mels=params.num_frequencies,
            sample_rate=params.sample_rate,
            f_min=params.min_frequency,
            f_max=params.max_frequency,
            n_stft=params.n_fft // 2 + 1,
            norm=params.mel_scale_norm,
            mel_scale=params.mel_scale_type,
        ).to(self.device)

        # https://pytorch.org/audio/stable/generated/torchaudio.transforms.InverseMelScale.html
        self.inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
            n_stft=params.n_fft // 2 + 1,
            n_mels=params.num_frequencies,
            sample_rate=params.sample_rate,
            f_min=params.min_frequency,
            f_max=params.max_frequency,
            max_iter=params.max_mel_iters,
            norm=params.mel_scale_norm,
            mel_scale=params.mel_scale_type,
        ).to(self.device)
    
    def spectrogram_from_audio(
        self,
        audio: pydub.AudioSegment
    ) -> np.ndarray:
        """
        Compute a spectrogram from an audio segment.

        Returns:
            spectrogram: (channel, frequency, time)
        """
        assert int(audio.frame_rate) == self.p.sample_rate, "Audio sample rate must match params"

        # Get the samples as a numpy array in (batch, samples) shape
        waveform = np.array([c.get_array_of_samples() for c in audio.split_to_mono()])

        # Convert to floats if necessary
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        waveform_tensor = torch.from_numpy(waveform).to(self.device)
        amplitudes_mel = self.mel_amplitudes_from_waveform(waveform_tensor)
        return amplitudes_mel.cpu().numpy()
    
    def audio_from_spectrogram(
        self,
        spectrogram: np.ndarray,
        apply_filters: bool = True,
    ) -> pydub.AudioSegment:
        """
        Reconstruct an audio segment from a spectrogram.
        Args:
            spectrogram: (batch, frequency, time)
            apply_filters: Post-process with normalization and compression
        Returns:
            audio: Audio segment with channels equal to the batch dimension
        """
        # Move to device
        amplitudes_mel = torch.from_numpy(spectrogram).to(self.device)

        # Reconstruct the waveform
        waveform = self.waveform_from_mel_amplitudes(amplitudes_mel)

        # Convert to audio segment
        segment = audio_util.audio_from_waveform(
            samples=waveform.cpu().numpy(),
            sample_rate=self.p.sample_rate,
            # Normalize the waveform to the range [-1, 1]
            normalize=True,
        )

        # Optionally apply post-processing filters
        # if apply_filters:
        #     segment = audio_util.apply_filters(
        #         segment,
        #         compression=False,
        #     )

        return segment

    def mel_amplitudes_from_waveform(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Torch-only function to compute Mel-scale amplitudes from a waveform.
        Args:
            waveform: (batch, samples)
        Returns:
            amplitudes_mel: (batch, frequency, time)
        """
        # Compute the complex-valued spectrogram
        spectrogram_complex = self.spectrogram_func(waveform)

        # Take the magnitude
        amplitudes = torch.abs(spectrogram_complex)

        # Convert to mel scale
        return self.mel_scaler(amplitudes)

    def waveform_from_mel_amplitudes(
        self,
        amplitudes_mel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Torch-only function to approximately reconstruct a waveform from Mel-scale amplitudes.
        Args:
            amplitudes_mel: (batch, frequency, time)
        Returns:
            waveform: (batch, samples)
        """
        # Convert from mel scale to linear
        amplitudes_linear = self.inverse_mel_scaler(amplitudes_mel)

        # Run the approximate algorithm to compute the phase and recover the waveform
        return self.inverse_spectrogram_func(amplitudes_linear)