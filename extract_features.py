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


# def calculate_pitch_features(pitches, sr):
#     if np.any(pitches > 0):
#         pitches = np.mean(pitches, axis=0)
#         epsilon = 1e-10
#         fo = np.mean(librosa.hz_to_midi(pitches[pitches > 0] + epsilon))
#         fhi = np.max(librosa.hz_to_midi(pitches + epsilon))
#         flo = np.min(librosa.hz_to_midi(pitches + epsilon))
#     else:
#         fo, fhi, flo = 0, 0, 0

#     return fo, fhi, flo
def calculate_pitch_features(pitches, sr):
    if np.any(pitches > 0):
        pitches = pitches[pitches > 0]  # Exclude negative pitch values
        fo = np.mean(librosa.hz_to_midi(pitches))
        fhi = np.max(librosa.hz_to_midi(pitches))
        flo = np.min(librosa.hz_to_midi(pitches))
    else:
        fo, fhi, flo = 0, 0, 0

    return fo, fhi, flo


# Create a function for each attribute calculation

def calculate_jitter(pitch_periods):
    jitter_values = np.abs(np.diff(pitch_periods))
    jitter_percentage = np.mean(jitter_values) * 100
    jitter_absolute = np.mean(jitter_values)
    rap = np.mean(np.abs(np.diff(pitch_periods, 2)))
    ppq = np.mean(np.abs(np.diff(pitch_periods, 3)))
    ddp = rap * 3
    return jitter_percentage, jitter_absolute, rap, ppq, ddp

def calculate_shimmer(amplitude_envelope):
    peaks, _ = find_peaks(amplitude_envelope)
    shimmer = np.mean(amplitude_envelope[peaks])
    shimmer_dB = max(0, 20 * np.log10(np.abs(shimmer)))  # Ensure non-negative value
    apq3 = np.mean(np.abs(np.diff(amplitude_envelope[peaks], 2)))
    apq5 = np.mean(np.abs(np.diff(amplitude_envelope[peaks], 4)))
    apq = np.mean(np.abs(np.diff(amplitude_envelope[peaks])))
    dda = np.mean(np.abs(np.diff(peaks)))
    return shimmer, shimmer_dB, apq3, apq5, apq, dda



def calculate_nhr_hnr(percussive, harmonic):
    nhr = np.mean(percussive / harmonic)
    hnr = np.mean(harmonic / percussive)
    return nhr, hnr


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
        # Extract only positive pitch values
        positive_pitches = pitches[pitches > 0]
        
        # Calculate spread1 as the standard deviation of MIDI pitch values
        if np.any(positive_pitches):
            epsilon = 1e-10  # Small epsilon to avoid division by zero
            midi_pitches = librosa.hz_to_midi(positive_pitches + epsilon)
            spread1_value = np.std(midi_pitches)
        else:
            spread1_value = 0
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

from scipy.stats import entropy

def calculate_ppe(audio_signal, fs):
    # Harmonic-Percussive source separation
    harmonic = librosa.effects.harmonic(audio_signal)

    # Pitch-related features
    pitches, magnitudes = librosa.core.piptrack(y=harmonic, sr=fs)

    # Get pitch periods
    pitch_periods = 1 / pitches[pitches > 0]

    # Filter out invalid pitch periods
    valid_pitch_periods = pitch_periods[pitch_periods > 0]

    # Calculate PPE using histogram and entropy
    if len(valid_pitch_periods) > 0:
        hist, bin_edges = np.histogram(valid_pitch_periods, bins='auto', density=True)
        ppe_value = entropy(hist)
    else:
        ppe_value = 0  # Set default value if there are no valid pitch periods

    return ppe_value



# Function to calculate mean and standard deviation for each feature
def calculate_mean_std(training_data):
    mean_values = {}
    std_values = {}

    # Extract feature values from training data
    feature_values = {feature: [sample[feature] for sample in training_data] for feature in training_data[0].keys()}

    # Calculate mean and standard deviation for each feature
    for feature, values in feature_values.items():
        mean_values[feature] = np.mean(values)
        std_values[feature] = np.std(values)

    return mean_values, std_values



# Function to standardize features
def standardize_features(features, mean_values, std_values):
    for feature, value in features.items():
        if std_values[feature] != 0:
            if isinstance(value, (int, float, np.float64)):
                features[feature] = (value - mean_values[feature]) / std_values[feature]
            elif isinstance(value, np.ndarray):
                features[feature] = (value - mean_values[feature]) / std_values[feature]
            else:
                print(f"Warning: Unsupported data type for feature {feature}")
        else:
            features[feature] = 0  # Handle the case where the standard deviation is zero

def extract_audio_features(audio_file_path):
    if not audio_file_path.lower().endswith(('.wav', '.mp3', '.ogg', '.flac', '.wma')):
        raise ValueError("Invalid file format. Please select a supported audio file.")

    # Load the audio file
    audio_signal, sr = librosa.load(audio_file_path, sr=None)

    # Feature extraction
    features = {}

    # Harmonic-Percussive source separation
    harmonic = librosa.effects.harmonic(audio_signal)

    # Calculate pitch-related features
    pitches, _ = librosa.core.piptrack(y=harmonic, sr=sr)
    fo, fhi, flo = calculate_pitch_features(pitches, sr)

    features['MDVP:Fo(Hz)'] = fo
    features['MDVP:Fhi(Hz)'] = fhi
    features['MDVP:Flo(Hz)'] = flo

    pitch_periods = 1 / pitches[pitches > 0]
    jitter_percentage, jitter_absolute, rap, ppq, ddp = calculate_jitter(pitch_periods)
    features['MDVP:Jitter(%)'] = jitter_percentage
    features['MDVP:Jitter(Abs)'] = jitter_absolute
    features['MDVP:RAP'] = rap
    features['MDVP:PPQ'] = ppq
    features['Jitter:DDP'] = ddp

    amplitude_envelope = np.abs(librosa.effects.preemphasis(harmonic))
    shimmer, shimmer_dB, apq3, apq5, apq, dda = calculate_shimmer(amplitude_envelope)
    features['MDVP:Shimmer'] = shimmer
    features['MDVP:Shimmer(dB)'] = shimmer_dB
    features['Shimmer:APQ3'] = apq3
    features['Shimmer:APQ5'] = apq5
    features['MDVP:APQ'] = apq
    features['Shimmer:DDA'] = dda

    percussive = librosa.effects.percussive(audio_signal)
    nhr, hnr = calculate_nhr_hnr(percussive, harmonic)
    features['NHR'] = nhr
    features['HNR'] = hnr

    # RPDE (Recurrence Period Density Entropy)
    try:
        features['RPDE'] = calculate_rpde(audio_signal)
    except Exception as e:
        print(f"Error calculating RPDE: {e}")
        features['RPDE'] = 0

    # D2 (Correlation dimension)
    try:
        features['D2'] = calculate_d2_parallelized(audio_signal)
    except Exception as e:
        print(f"Error calculating D2: {e}")
        features['D2'] = 0

    # DFA (Detrended Fluctuation Analysis)
    try:
        features['DFA'] = calculate_dfa(audio_signal)
    except Exception as e:
        print(f"Error calculating DFA: {e}")
        features['DFA'] = 0

    # Calculate spread1
    try:
        features['spread1'] = calculate_spread1(audio_signal, sr)
    except Exception as e:
        print(f"Error calculating spread1: {e}")
        features['spread1'] = 0

    # Calculate spread2
    try:
        features['spread2'] = calculate_spread2(audio_signal)
    except Exception as e:
        print(f"Error calculating spread2: {e}")
        features['spread2'] = 0

    # Calculate PPE
    try:
        features['PPE'] = calculate_ppe(audio_signal, sr)
    except Exception as e:
        print(f"Error calculating PPE: {e}")
        features['PPE'] = 0


    return features

def calculate_mean_std(features):
    all_values = np.array(list(features.values()))
    mean_values = np.mean(all_values, axis=0)
    std_values = np.std(all_values, axis=0)
    return mean_values, std_values

if __name__ == "__main__":
    try:
        # Example usage
        audio_file_path = "recorded_audio.wav"
        features = extract_audio_features(audio_file_path)

        # Calculate mean and std values
        mean_values, std_values = calculate_mean_std(features)

        # Standardize features
        standardized_features = {}
        for feature, value in features.items():
            standardized_features[feature] = (value - mean_values) / std_values

        # Display the standardized features
        logging.info("\nStandardized Features:")
        for feature, value in standardized_features.items():
            logging.info(f"{feature}: {value}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")