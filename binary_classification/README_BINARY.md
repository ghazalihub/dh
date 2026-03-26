# Ortho AI - Binary Classification (Normal vs Abnormal)

This folder contains a classification model for the binary task: **Normal vs Abnormal**.

## Normalization
- Patients with 'Disk Hernia' or 'Spondylolisthesis' are mapped to 'Abnormal'.
- Patients with 'Normal' status are kept as 'Normal'.

## Files
- `train_binary.py`: Script to train and tune multiple models for binary classification.
- `best_binary_model.pkl`: Serialized best model (currently SVM).
- `processed_data_binary.pkl`: Preprocessing tools (Scaler, LabelEncoder).
- `binary_evaluation_report.txt`: Performance results for the binary model.
- `ortho_ai_binary.py`: Command-line interface for predictions.

## Usage

### Training
```bash
python binary_classification/train_binary.py
```

### Prediction
```bash
python binary_classification/ortho_ai_binary.py <pelvic_incidence> <pelvic_tilt> <lumbar_lordosis_angle> <sacral_slope> <pelvic_radius> <degree_spondylolisthesis>
```

## Results
The binary classification model (SVM) achieved an accuracy of **90.32%**, which is significantly higher than the 3-class classification models. This demonstrates that identifying abnormality is a more robust task for this dataset.
