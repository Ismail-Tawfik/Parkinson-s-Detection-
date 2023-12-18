
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
    new_data_example = np.array([[187.73300,202.32400,173.01500,0.00316,0.00002,0.00168,0.00182,0.00504,0.01663,0.15100,0.00829,0.01003,0.01366,0.02488,0.00265,26.31000,0.396793,0.758324,-6.006647,0.266892,2.382544,0.160691]])
    predictions_example = make_prediction(new_data_example, loaded_bc_model, loaded_scaler)
    print("Predictions:", predictions_example)
