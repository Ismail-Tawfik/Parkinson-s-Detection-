import logging
from record_audio import record_audio, save_audio_to_file
from extract_features import extract_audio_features
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

        loaded_bc_model, loaded_scaler = load_models()
        predictions = make_prediction(np.array([list(features.values())]), loaded_bc_model, loaded_scaler)

        logging.info("\nPrediction:")
        logging.info(f"Prediction: {predictions}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
