# Ortho AI - LightGBM Experiment

This folder contains an experimental classification model using **LightGBM**.

## Files
- `train_lgbm.py`: Script to train and tune the LightGBM model.
- `best_lgbm_model.pkl`: Serialized best LightGBM model.
- `processed_data_lgbm.pkl`: Preprocessing tools (Scaler, LabelEncoder).
- `lgbm_evaluation_report.txt`: Performance results for the LightGBM model.
- `ortho_ai_lgbm.py`: Command-line interface for predictions using LightGBM.

## Usage

### Training
```bash
python lightgbm_experiment/train_lgbm.py
```

### Prediction
```bash
python lightgbm_experiment/ortho_ai_lgbm.py <pelvic_incidence> <pelvic_tilt> <lumbar_lordosis_angle> <sacral_slope> <pelvic_radius> <degree_spondylolisthesis>
```

## Results
The LightGBM model achieved an accuracy of **82.26%**, which is lower than the Random Forest model's **87.1%**. For this particular dataset size and characteristics, Random Forest appears to be a more suitable choice.
