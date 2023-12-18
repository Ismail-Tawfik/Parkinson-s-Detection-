import pickle
import numpy as np

def load_models():
    model_names = ["lr_model.pkl", "nb_model.pkl", "svm_model.pkl", "dt_model.pkl",
                   "abc_model.pkl", "rf_model.pkl", "knn_model.pkl", "bc_model.pkl", "XG_model.pkl"]

    # Load the scaler
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    loaded_models = []

    for model_name in model_names:
        with open(model_name, "rb") as file:
            loaded_model = pickle.load(file)

        # Set the scaler attribute directly on the loaded model
        loaded_model.scaler = scaler
        loaded_models.append(loaded_model)

    return loaded_models

def make_prediction(new_data, loaded_model_and_scaler):
    loaded_model = loaded_model_and_scaler

    if hasattr(loaded_model, 'scaler'):
        loaded_scaler = loaded_model.scaler
    else:
        loaded_scaler = None

    # Scale the new data using the loaded scaler if available
    new_data_scaled = loaded_scaler.transform(new_data) if loaded_scaler else new_data
    # Make predictions using the loaded model
    predictions = loaded_model.predict(new_data_scaled)
    print(f"Predictions for {type(loaded_model).__name__}: {predictions}")
    return predictions

# Example usage:
if __name__ == "__main__":
    loaded_models_and_scalers = load_models()
    

    # Test Parkinson case 
    new_data_example = np.array([[ 1.187470e+02,  1.237230e+02,  1.098360e+02,  3.310000e-03,
        3.000000e-05,  1.680000e-03,  1.710000e-03,  5.040000e-03,
        1.043000e-02,  9.900000e-02,  4.900000e-03,  6.210000e-03,
        9.030000e-03,  1.471000e-02,  5.040000e-03,  2.561900e+01,
        4.822960e-01,  7.230960e-01, -6.448134e+00,  1.787130e-01,
        2.034827e+00,  1.414220e-01]])


    # Test health case 
    # new_data_example = np.array([[ 2.42852e+02,  2.55034e+02,  2.27911e+02,  2.25000e-03,
    #     9.00000e-06,  1.17000e-03,  1.39000e-03,  3.50000e-03,
    #     1.49400e-02,  1.34000e-01,  8.47000e-03,  8.79000e-03,
    #     1.01400e-02,  2.54200e-02,  4.76000e-03,  2.50320e+01,
    #     4.31285e-01,  6.38928e-01, -6.99582e+00,  1.02083e-01,
    #     2.36580e+00,  1.02706e-01]])

    for loaded_model_and_scaler in loaded_models_and_scalers:
        predictions_example = make_prediction(new_data_example, loaded_model_and_scaler)
