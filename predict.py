
import pickle
import numpy as np

def load_models():
    # Load the Bagging Classifier model
    with open("BC_model.pkl", "rb") as file:
        loaded_bc_model = pickle.load(file)

    # Load the Standard Scaler
    with open("scaler.pkl", "rb") as file:
        loaded_scaler = pickle.load(file)

    return loaded_bc_model, loaded_scaler

def make_prediction(new_data, loaded_bc_model, loaded_scaler):
    # Scale the new data using the loaded scaler
    new_data_scaled = loaded_scaler.transform(new_data)

    # Make predictions using the loaded model
    predictions = loaded_bc_model.predict(new_data_scaled)

    return predictions

# # Example usage:
if __name__ == "__main__":
    loaded_bc_model, loaded_scaler = load_models()

    new_data_example = np.array([[148.46200,161.07800,141.99800,0.00397,0.00003,0.00202,0.00235,0.00605,0.01831,0.16300,0.00950,0.01103,0.01559,0.02849,0.00639,22.86600,0.408598,0.768845,-5.704053,0.216204,2.679185,0.197710]])

    predictions_example = make_prediction(new_data_example, loaded_bc_model, loaded_scaler)
    print("Predictions:", predictions_example)
