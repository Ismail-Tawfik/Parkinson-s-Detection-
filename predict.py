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
    predictions = loaded_model.predict(new_data)
    print(f"Predictions for {type(loaded_model).__name__}: {predictions}")
    return predictions


# Example usage:
if __name__ == "__main__":
    loaded_models_and_scalers = load_models()


    # Ismail
    new_data_example = np.array([[
    2.2878668133030007, 3.4660797412225013, 1.3582685779732127,
    -0.4337309900580823, -0.43383884410645884, -0.4338381487104706,
    -0.43383676654098996, -0.43383457904882994, -0.43383314400570877,
    -0.43383993354129097, -0.4338348768217214, -0.4338234698333895,
    -0.4338368911645815, -0.2574718438735811, -0.3599132077985072,
    -0.2420725170448237, -0.40178432059638414, -0.3553701324503661,
    -0.4301046058294808, 0.03581518394011348, -0.43382706633007867,
    -0.32923897868408286]])

#     # mohamed aly kly
#     new_data_example = np.array([[
#     2.568472253999617, 3.2706023386723273, 1.2800813391298682,
#     -0.4083985120847869, -0.4084647172831394, -0.4084642621208181,
#     -0.4084633686362775, -0.4084620143174211, -0.4082687275138612,
#     -0.40846538602251664, -0.408308933729791, -0.40796110170589506,
#     -0.408369774069016, -0.27292318264438886, -0.7047994864488117,
#     -0.3453398729718495, -0.3819982135812047, -0.3484672714473056,
#     -0.4065835207711818, 0.13561756459130778, -0.4083026206052499,
#     -0.30273253043960485
# ]])
    

    # Test Parkinson case 
    # new_data_example = np.array([[ 1.187470e+02,  1.237230e+02,  1.098360e+02,  3.310000e-03,
    #     3.000000e-05,  1.680000e-03,  1.710000e-03,  5.040000e-03,
    #     1.043000e-02,  9.900000e-02,  4.900000e-03,  6.210000e-03,
    #     9.030000e-03,  1.471000e-02,  5.040000e-03,  2.561900e+01,
    #     4.822960e-01,  7.230960e-01, -6.448134e+00,  1.787130e-01,
    #     2.034827e+00,  1.414220e-01]])
    
    # Test health case 
    # new_data_example = np.array([[ 2.42852e+02,  2.55034e+02,  2.27911e+02,  2.25000e-03,
    #     9.00000e-06,  1.17000e-03,  1.39000e-03,  3.50000e-03,
    #     1.49400e-02,  1.34000e-01,  8.47000e-03,  8.79000e-03,
    #     1.01400e-02,  2.54200e-02,  4.76000e-03,  2.50320e+01,
    #     4.31285e-01,  6.38928e-01, -6.99582e+00,  1.02083e-01,
    #     2.36580e+00,  1.02706e-01]])

    all_predictions = []
    for loaded_model_and_scaler in loaded_models_and_scalers:
        predictions_example = make_prediction(new_data_example, loaded_model_and_scaler)
        all_predictions.extend(predictions_example)

    average_prediction = np.mean(all_predictions)
    print("\nAverage Prediction:")
    print(average_prediction)
