import os
import numpy as np
import librosa
from scipy.signal import find_peaks
from scipy.io.wavfile import read
import logging
import nolds
from pyrpde import rpde
from scipy.io.wavfile import read
from scipy.stats import entropy
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_rpde(signal):
    try:
        # Assuming 'signal' is a 1D array of audio data
        entropy, histogram = rpde(signal, tau=30, dim=4, epsilon=0.01, tmax=1500)
        return entropy
    except Exception as e:
        print(f"Error calculating RPDE: {e}")
        return 0  # Set default value in case of an error
    

# def calculate_d2(signal, emb_dim=10):
#     try:
#         # Calculate D2 using nolds library
#         d2_value = nolds.corr_dim(signal, emb_dim=emb_dim)
#         return d2_value
#     except Exception as e:
#         print(f"Error calculating D2: {e}")
#         return 0  # Set default value in case of an error
def calculate_d2_parallel(signal, emb_dim=10):
    try:
        # Calculate D2 using nolds library
        d2_value = nolds.corr_dim(signal, emb_dim=emb_dim)
        return d2_value
    except Exception as e:
        print(f"Error calculating D2: {e}")
        return 0  # Set default value in case of an error

def calculate_d2_parallelized(audio_signal, segment_size=10000, num_processes=2):
    segments = [audio_signal[i:i+segment_size] for i in range(0, len(audio_signal), segment_size)]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(calculate_d2_parallel, segment) for segment in segments]
        results = [future.result() for future in futures]

    return np.mean(results)



def calculate_dfa(signal):
    try:
        # Calculate DFA using nolds library
        dfa_value = nolds.dfa(signal)
        return dfa_value
    except Exception as e:
        print(f"Error calculating DFA: {e}")
        return 0  # Set default value in case of an error

def calculate_spread1(audio_signal, fs):
    # Harmonic-Percussive source separation
    harmonic = librosa.effects.harmonic(audio_signal)

    # Pitch-related features
    pitches, magnitudes = librosa.core.piptrack(y=harmonic, sr=fs)

    # Handle the case when there are no valid pitch values
    if np.any(pitches > 0):
        pitches = np.mean(pitches, axis=0)
        epsilon = 1e-10  # Small epsilon to avoid division by zero
        midi_pitches = librosa.hz_to_midi(pitches[pitches > 0] + epsilon)
        
        # Calculate spread1 as the standard deviation of MIDI pitch values
        spread1_value = np.std(midi_pitches)
    else:
        # Set default value if there are no valid pitch values
        spread1_value = 0

    return spread1_value

def calculate_spread2(audio_signal):
    # Harmonic-Percussive source separation
    harmonic = librosa.effects.harmonic(audio_signal)

    # Find peaks in the amplitude envelope of the harmonic component
    amplitude_envelope = np.abs(librosa.effects.preemphasis(harmonic))
    peaks, _ = find_peaks(amplitude_envelope)

    # Calculate spread2 as the standard deviation of peak amplitudes
    spread2_value = np.std(amplitude_envelope[peaks])

    return spread2_value

def calculate_ppe(audio_signal, fs):
    # Harmonic-Percussive source separation
    harmonic = librosa.effects.harmonic(audio_signal)

    # Pitch-related features
    pitches, magnitudes = librosa.core.piptrack(y=harmonic, sr=fs)

    # Get pitch periods
    pitch_periods = 1 / pitches[pitches > 0]

    # Filter out invalid pitch periods
    valid_pitch_periods = pitch_periods[pitch_periods > 0]

    # Calculate PPE using entropy
    if len(valid_pitch_periods) > 0:
        ppe_value = entropy(valid_pitch_periods)
    else:
        ppe_value = 0  # Set default value if there are no valid pitch periods

    return ppe_value

def extract_audio_features(audio_file_path):
    if not audio_file_path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.wma')):
        raise ValueError("Invalid file format. Please select a supported audio file.")

    # Load the audio file
    audio_signal, sr = librosa.load(audio_file_path, sr=None)

    # Feature extraction
    features = {}

    # Harmonic-Percussive source separation
    harmonic = librosa.effects.harmonic(audio_signal)

    # Pitch-related features
    pitches, magnitudes = librosa.core.piptrack(y=harmonic, sr=sr)

    # Handle the case when there are no valid pitch values
    if np.any(pitches > 0):
        pitches = np.mean(pitches, axis=0)
        epsilon = 1e-10  # Small epsilon to avoid division by zero
        features['MDVP:Fo(Hz)'] = np.mean(librosa.hz_to_midi(pitches[pitches > 0] + epsilon))
        features['MDVP:Fhi(Hz)'] = np.max(librosa.hz_to_midi(pitches + epsilon))
        features['MDVP:Flo(Hz)'] = np.min(librosa.hz_to_midi(pitches + epsilon))
    else:
        # Set default values if there are no valid pitch values
        features['MDVP:Fo(Hz)'] = 0
        features['MDVP:Fhi(Hz)'] = 0
        features['MDVP:Flo(Hz)'] = 0

    # Jitter-related features
    pitch_periods = 1 / pitches[pitches > 0]
    jitter_values = np.abs(np.diff(pitch_periods))
    features['MDVP:Jitter(%)'] = np.mean(jitter_values) * 100  # Convert to percentage
    features['MDVP:Jitter(Abs)'] = np.mean(jitter_values)
    features['MDVP:RAP'] = np.mean(np.abs(np.diff(pitch_periods, 2)))
    features['MDVP:PPQ'] = np.mean(np.abs(np.diff(pitch_periods, 3)))
    features['Jitter:DDP'] = features['MDVP:RAP'] * 3  # Assuming DDP is 3 times RAP

    # Shimmer-related features (using amplitude-based shimmer)
    amplitude_envelope = np.abs(librosa.effects.preemphasis(harmonic))
    peaks, _ = find_peaks(amplitude_envelope)
    features['MDVP:Shimmer'] = np.mean(amplitude_envelope[peaks])
    features['MDVP:Shimmer(dB)'] = 20 * np.log10(features['MDVP:Shimmer'])  # Convert to dB

    # Shimmer:APQ3 and Shimmer:APQ5
    features['Shimmer:APQ3'] = np.mean(np.abs(np.diff(amplitude_envelope[peaks], 2)))
    features['Shimmer:APQ5'] = np.mean(np.abs(np.diff(amplitude_envelope[peaks], 4)))

    # MDVP:APQ
    features['MDVP:APQ'] = np.mean(np.abs(np.diff(amplitude_envelope[peaks])))

    # Shimmer:DDA
    features['Shimmer:DDA'] = np.mean(np.abs(np.diff(peaks)))

    # NHR (Noise-to-Harmonics Ratio)
    percussive = librosa.effects.percussive(audio_signal)
    features['NHR'] = np.mean(percussive / harmonic)

    # HNR (Harmonics-to-Noise Ratio)
    features['HNR'] = np.mean(harmonic / percussive)

    # RPDE (Recurrence Period Density Entropy)
    try:
        print("RPDE")
        features['RPDE'] = calculate_rpde(audio_signal)
    except Exception as e:
        print(f"Error calculating RPDE: {e}")
        features['RPDE'] = 0  # Set default value in case of an error

    # D2 (Correlation dimension)
    # try:
    #     print(4444444)
    #     features['D2'] = calculate_d2(audio_signal)
    # except Exception as e:
    #     print(f"Error calculating D2: {e}")
    #     features['D2'] = 0  # Set default value in case of an error
    # D2 (Correlation dimension)
    try:
        print("Calculating D2 in parallel...")
        features['D2'] = calculate_d2_parallelized(audio_signal)
    except Exception as e:
        print(f"Error calculating D2: {e}")
        features['D2'] = 0  # Set default value in case of an error
    # DFA (Detrended Fluctuation Analysis)
    try:
        print("33333")
        features['DFA'] = calculate_dfa(audio_signal)
    except Exception as e:
        print(f"Error calculating DFA: {e}")
        features['DFA'] = 0  # Set default value in case of an error

    # Calculate spread1
    try:
        features['spread1'] = calculate_spread1(audio_signal, sr)
    except Exception as e:
        print(f"Error calculating spread1: {e}")
        features['spread1'] = 0  # Set default value in case of an error
    # Calculate spread2
    try:
        features['spread2'] = calculate_spread2(audio_signal)
    except Exception as e:
        print(f"Error calculating spread2: {e}")
        features['spread2'] = 0  # Set default value in case of an error
    # Calculate PPE
    try:
        features['PPE'] = calculate_ppe(audio_signal, sr)
    except Exception as e:
        print(f"Error calculating PPE: {e}")
        features['PPE'] = 0  # Set default value in case of an error
    # Add the 'status' feature
    # features['status'] = 0  # Set to 0 for no Parkinson's

    return features

if __name__ == "__main__":
    try:
        # Example usage
        audio_file_path = "recorded_audio.wav"
        features = extract_audio_features(audio_file_path)

        # Display the extracted features
        logging.info("\nExtracted Features:")
        for feature, value in features.items():
            logging.info(f"{feature}: {value}")


    except Exception as e:
        logging.error(f"An error occurred: {e}")
