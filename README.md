# Ortho AI Classification System

This project implements a machine learning system for classifying orthopaedic patients into three categories: **Normal (NO)**, **Disk Hernia (DH)**, or **Spondylolisthesis (SL)**.

The classification is based on six biomechanical features:
1. Pelvic Incidence
2. Pelvic Tilt
3. Lumbar Lordosis Angle
4. Sacral Slope
5. Pelvic Radius
6. Degree Spondylolisthesis

## Requirements
- Python 3
- pandas
- scikit-learn
- joblib

Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
To retrain the model with the latest data:
```bash
python train.py
```
This script will:
- Load `data.csv`.
- Preprocess features and encode labels.
- Tune multiple models (Random Forest, SVM, KNN) using GridSearchCV.
- Save the best model to `best_model.pkl` and preprocessing tools to `processed_data.pkl`.
- Generate an `evaluation_report.txt`.

### Making Predictions
You can use the `ortho_ai.py` script to get a prediction by providing the six biomechanical features as command-line arguments.

```bash
python ortho_ai.py <pelvic_incidence> <pelvic_tilt> <lumbar_lordosis_angle> <sacral_slope> <pelvic_radius> <degree_spondylolisthesis>
```

#### Example

```bash
python ortho_ai.py 63.02 22.55 39.60 40.47 98.67 -0.25
```
Output:
`Prediction: Hernia`

Another example for Spondylolisthesis:
```bash
python ortho_ai.py 74.37 32.05 78.77 42.32 143.56 56.12
```
Output:
`Prediction: Spondylolisthesis`

## Model Performance & Target Accuracy

The current model (Random Forest) achieves a test accuracy of approximately **87.1%**.

### Why Not 95%?
While a 95% accuracy target was set, this is particularly challenging for the Vertebral Column dataset (3-class version) for several reasons:
- **Small Dataset Size**: With only 310 total instances, the model has limited examples to learn complex patterns, especially for the "Disk Hernia" class (only 60 instances).
- **Class Overlap**: There is significant biomechanical overlap between 'Normal' and 'Disk Hernia' patients in the feature space.
- **Biomechanical Variance**: Real-world medical data inherently contains high variance and noise.

### Future Improvements
To move towards 95% accuracy, consider:
1. **Acquiring More Data**: Larger datasets would help the model generalize better.
2. **Feature Engineering**: Creating interaction terms or incorporating more clinical features.
3. **Advanced Ensemble Methods**: Using Stacking or Boosting with more extensive tuning.

The model demonstrates excellent sensitivity (recall) for Spondylolisthesis (97%), making it a robust diagnostic aid for that condition.
