import joblib
import pandas as pd
import sys
import numpy as np

def predict_ortho(data_dict):
    """
    data_dict: dict containing the 6 biomechanical features
    """
    # Load model and preprocessing tools
    try:
        data_processed = joblib.load('processed_data.pkl')
        model = joblib.load('best_model.pkl')
        scaler = data_processed['scaler']
        le = data_processed['label_encoder']
    except FileNotFoundError:
        print("Model files not found. Please run the training steps first.")
        return None

    # Define feature order
    features_order = [
        'pelvic_incidence',
        'pelvic_tilt',
        'lumbar_lordosis_angle',
        'sacral_slope',
        'pelvic_radius',
        'degree_spondylolisthesis'
    ]

    # Prepare input as DataFrame for scaler
    input_data = pd.DataFrame([data_dict])[features_order]

    # Scale features
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction_encoded = model.predict(input_scaled)
    prediction_label = le.inverse_transform(prediction_encoded)

    return prediction_label[0]

def main():
    if len(sys.argv) == 7:
        try:
            features = [float(x) for x in sys.argv[1:7]]
            data = {
                'pelvic_incidence': features[0],
                'pelvic_tilt': features[1],
                'lumbar_lordosis_angle': features[2],
                'sacral_slope': features[3],
                'pelvic_radius': features[4],
                'degree_spondylolisthesis': features[5]
            }
            result = predict_ortho(data)
            print(f"Prediction: {result}")
        except ValueError:
            print("Invalid input: Please provide 6 numeric biomechanical features.")
    else:
        print("Usage: python ortho_ai.py <pelvic_incidence> <pelvic_tilt> <lumbar_lordosis_angle> <sacral_slope> <pelvic_radius> <degree_spondylolisthesis>")
        print("Example: python ortho_ai.py 63.02 22.55 39.60 40.47 98.67 -0.25")

if __name__ == "__main__":
    main()
