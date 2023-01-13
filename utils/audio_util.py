"""
Audio utility functions.
"""

import io

import numpy as np
import pydub
from scipy.io import wavfile


def audio_from_waveform(
    samples: np.ndarray, sample_rate: int, normalize: bool = False
) -> pydub.AudioSegment:
    """
    Convert a numpy array of samples of a waveform to an audio segment.
    Args:
        samples: (channels, samples) array
    """
    # Normalize volume to fit in int16
    if normalize:
        samples *= np.iinfo(np.int16).max / np.max(np.abs(samples))

    # Transpose and convert to int16
    samples = samples.transpose(1, 0)
    samples = samples.astype(np.int16)

    # Write to the bytes of a WAV file
    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples)
    wav_bytes.seek(0)

    # Read into pydub
    return pydub.AudioSegment.from_wav(wav_bytes)