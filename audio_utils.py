# audio_utils.py
import librosa
import numpy as np
import soundfile as sf
import io
from typing import Tuple, Optional

TARGET_SAMPLE_RATE = 16000  

def load_audio(
    audio_path: str, 
    target_sr: int = TARGET_SAMPLE_RATE,
    force_mono: bool = False
) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file, resamples it, and optionally converts to mono.
    
    Returns:
        waveform (np.ndarray): The audio waveform.
        sample_rate (int): The sample rate of the loaded audio.
    """
    try:
        # mono=False preserves the original channel count
        waveform, sample_rate = librosa.load(audio_path, sr=target_sr, mono=force_mono)
        return waveform, sample_rate
        
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return np.array([]), 0

def get_single_audio_waveform(
    audio_path: str, 
    target_sr: int = TARGET_SAMPLE_RATE
) -> Optional[np.ndarray]:
    """
    Loads an audio file for single-mode evaluation.
    Per user request, this DOES NOT force mono.
    The processor is expected to handle stereo input.
    """
    waveform, sample_rate = load_audio(audio_path, target_sr=target_sr, force_mono=False)
    
    if waveform.size == 0:
        return None
        
    return waveform

def get_audio_channels(
    audio_path: str, 
    target_sr: int = TARGET_SAMPLE_RATE
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads an audio file and returns its left and right channels as separate
    mono waveforms. If the audio is mono, it returns the mono waveform
    for both left and right. This is for 'double' modes.
    """
    # Here we still load with mono=False to get channel info
    waveform, sample_rate = load_audio(audio_path, target_sr=target_sr, force_mono=False)
    
    if waveform.size == 0:
        return None, None
        
    if waveform.ndim == 1:
        # It's already mono, return it for both channels
        return waveform, waveform
    elif waveform.ndim == 2 and waveform.shape[0] >= 2:
        # It's stereo, return left (channel 0) and right (channel 1)
        left_channel = waveform[0]
        right_channel = waveform[1]
        return left_channel, right_channel
    else:
        print(f"Warning: Audio {audio_path} has unexpected shape {waveform.shape}. Treating as mono.")
        # Fallback to mono
        mono_wave = np.mean(waveform, axis=0) if waveform.ndim > 1 else waveform
        return mono_wave, mono_wave

def convert_to_wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """
    Converts a numpy audio waveform (mono or stereo) to a WAV file in memory.
    """
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer.read()

def encode_audio_to_base64(waveform: np.ndarray, sample_rate: int) -> str:
    """
    Converts a numpy audio waveform (mono or stereo) to a base64-encoded string.
    """
    import base64
    wav_bytes = convert_to_wav_bytes(waveform, sample_rate)
    return base64.b64encode(wav_bytes).decode('utf-8')

def read_file_as_base64(audio_path: str) -> str:
    """
    Reads an audio file directly from disk and encodes it as base64.
    This is for APIs like GPT-4o that take the raw file.
    """
    import base64
    with open(audio_path, "rb") as f:
        wav_data = f.read()
    return base64.b64encode(wav_data).decode('utf-8')