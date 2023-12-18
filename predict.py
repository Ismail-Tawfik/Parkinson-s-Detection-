import pickle
import numpy as np

def load_models():
    model_names = ["lr_model.pkl", "nb_model.pkl", "svm_model.pkl", "dt_model.pkl",
                   "abc_model.pkl", "rf_model.pkl", "knn_model.pkl", "bc_model.pkl"]
    
    loaded_models = []

    for model_name in model_names:
        with open(model_name, "rb") as file:
            loaded_model = pickle.load(file)
            # Assuming that the model has a scaler attribute
            scaler = getattr(loaded_model, 'scaler', None)
            loaded_models.append((loaded_model, scaler))

    return loaded_models

def make_prediction(new_data, loaded_model_and_scaler):
    loaded_model, loaded_scaler = loaded_model_and_scaler
    # Scale the new data using the loaded scaler if available
    new_data_scaled = loaded_scaler.transform(new_data) if loaded_scaler else new_data
    # Make predictions using the loaded model
    predictions = loaded_model.predict(new_data_scaled)
    print(f"Predictions for {type(loaded_model).__name__}: {predictions}")
    return predictions

# Example usage:
if __name__ == "__main__":
    loaded_models_and_scalers = load_models()
    new_data_example = np.array([[-5.84858231e-01, -7.23571195e-01,  7.15329331e-02,
        -1.21990710e-01, -1.36165869e-01, -4.30357898e-01,
        -4.76112596e-01, -4.30053499e-01, -7.62012217e-01,
        -7.79508995e-01, -7.19947312e-01, -9.04951931e-01,
        -7.21091736e-01, -7.19670689e-01, -2.93878202e-01,
         2.13565509e-01,  5.84178751e-01,  2.65430723e-01,
        -6.29692850e-01,  7.96178128e-01, -8.02449343e-01,
        -7.62828781e-01]])

    for loaded_model_and_scaler in loaded_models_and_scalers:
        predictions_example = make_prediction(new_data_example, loaded_model_and_scaler)
