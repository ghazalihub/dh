import joblib
import pandas as pd
import sys
import os

def predict_ortho_lgbm(data_dict):
    """
    data_dict: dict containing the 6 biomechanical features
    """
    # Use absolute path for loading model files relative to this script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_path = os.path.join(base_dir, 'processed_data_lgbm.pkl')
    best_model_path = os.path.join(base_dir, 'best_lgbm_model.pkl')

    try:
        data_processed = joblib.load(processed_data_path)
        model = joblib.load(best_model_path)
        scaler = data_processed['scaler']
        le = data_processed['label_encoder']
    except FileNotFoundError:
        print(f"LightGBM model files not found. Please ensure 'best_lgbm_model.pkl' and 'processed_data_lgbm.pkl' are in {base_dir}")
        return None

    features_order = [
        'pelvic_incidence',
        'pelvic_tilt',
        'lumbar_lordosis_angle',
        'sacral_slope',
        'pelvic_radius',
        'degree_spondylolisthesis'
    ]

    input_data = pd.DataFrame([data_dict])[features_order]
    input_scaled = pd.DataFrame(scaler.transform(input_data), columns=features_order)

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
            result = predict_ortho_lgbm(data)
            print(f"Prediction (LightGBM): {result}")
        except ValueError:
            print("Invalid input: Please provide 6 numeric biomechanical features.")
    else:
        print("Usage: python lightgbm_experiment/ortho_ai_lgbm.py <pelvic_incidence> <pelvic_tilt> <lumbar_lordosis_angle> <sacral_slope> <pelvic_radius> <degree_spondylolisthesis>")

if __name__ == "__main__":
    main()
