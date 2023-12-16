import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as write_wav
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def record_audio(duration=10, fs=44100):
    try:
        logger.info(f"Recording audio for {duration} seconds...")
        recording = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype=np.float32)
        sd.wait()

        recording_int16 = (np.iinfo(np.int16).max * recording).astype(np.int16)
        return recording_int16.flatten()

    except Exception as e:
        logger.error(f"Error recording audio: {e}")
        raise RuntimeError(f"Error recording audio: {e}")

def save_audio_to_file(audio_data, filename, fs=44100):
    try:
        write_wav(filename, fs, audio_data)
        logger.info(f"Audio saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving audio to {filename}: {e}")
        raise RuntimeError(f"Error saving audio to {filename}: {e}")
