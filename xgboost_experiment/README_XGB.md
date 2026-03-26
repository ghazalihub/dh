# Ortho AI - XGBoost Experiment

This folder contains an experimental classification model using **XGBoost**.

## Files
- `train_xgb.py`: Script to train and tune the XGBoost model.
- `best_xgb_model.pkl`: Serialized best XGBoost model.
- `processed_data_xgb.pkl`: Preprocessing tools (Scaler, LabelEncoder).
- `xgb_evaluation_report.txt`: Performance results for the XGBoost model.
- `ortho_ai_xgb.py`: Command-line interface for predictions using XGBoost.

## Usage

### Training
```bash
python xgboost_experiment/train_xgb.py
```

### Prediction
```bash
python xgboost_experiment/ortho_ai_xgb.py <pelvic_incidence> <pelvic_tilt> <lumbar_lordosis_angle> <sacral_slope> <pelvic_radius> <degree_spondylolisthesis>
```

## Results
The XGBoost model achieved an accuracy of **80.65%**, matching the LightGBM experiment but remaining below the Random Forest's **87.1%**.
