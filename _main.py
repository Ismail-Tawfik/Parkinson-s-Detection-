import logging
from record_audio import record_audio, save_audio_to_file
from extract_features import extract_audio_features, calculate_mean_std, standardize_features
from predict import load_models, make_prediction
import numpy as np
from tkinter import Tk, filedialog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    fs = 44100  # Sampling frequency
    duration = 10  # Recording duration in seconds

    try:
        user_choice = input("Do you want to record audio or choose a file? (record/file): ").lower()

        if user_choice == 'record':
            audio_data = record_audio(duration, fs)
            filename = "recorded_audio.wav"
            save_audio_to_file(audio_data, filename, fs)
            logging.info(f"Audio saved to {filename}")
        elif user_choice == 'file':
            root = Tk()
            root.withdraw()
            filename = filedialog.askopenfilename(title="Select an audio file")
            logging.info(f"Selected audio file: {filename}")
        else:
            logging.warning("Invalid choice. Exiting.")
            return

        features = extract_audio_features(filename)

        logging.info("\nExtracted Features:")
        for feature, value in features.items():
            logging.info(f"{feature}: {value}")

        # Calculate mean and std values
        mean_values, std_values = calculate_mean_std(features)

        # Standardize features
        standardized_features = {}
        for feature, value in features.items():
            standardized_features[feature] = (value - mean_values) / std_values
        logging.info("\nStandardized Features:")
        for feature, value in standardized_features.items():
            logging.info(f"{feature}: {value}")

        loaded_models = load_models()

        predictions_list = []

        for model_and_scaler in loaded_models:
            predictions = make_prediction(np.array([list(standardized_features.values())]), model_and_scaler)
            predictions_list.append(predictions)

        logging.info("\nPredictions:")
        for i, (model_and_scaler, predictions) in enumerate(zip(loaded_models, predictions_list)):
            logging.info(f"Predictions for {type(model_and_scaler).__name__}: {predictions}")

        # Calculate majority vote
        final_prediction = int(np.mean(predictions_list) >= 0.5)

        logging.info("\nMajority Vote (Final Prediction):")
        logging.info(final_prediction)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()