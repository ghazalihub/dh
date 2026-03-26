import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_binary_rf():
    print("Loading data for Binary Classification (Normal vs Abnormal)...")
    df = pd.read_csv('data.csv')

    # Map classes to binary: Hernia and Spondylolisthesis to 'Abnormal', keep 'Normal'
    df['class_binary'] = df['class'].replace(['Hernia', 'Spondylolisthesis'], 'Abnormal')

    # Encode labels
    le = LabelEncoder()
    df['class_encoded'] = le.fit_transform(df['class_binary'])
    print(f"Classes: {le.classes_}") # ['Abnormal', 'Normal']

    # Features and target
    X = df.drop(['class', 'class_binary', 'class_encoded'], axis=1)
    y = df['class_encoded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    print("\nTuning Random Forest for Binary Classification...")
    rf_model = RandomForestClassifier(random_state=42)

    # Controlled param grid to avoid timeout
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None]
    }

    grid = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    print(f"  Best CV score: {grid.best_score_:.4f}")
    best_rf = grid.best_estimator_

    y_pred = best_rf.predict(X_test_scaled)
    test_score = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {test_score:.4f}")

    # Save files
    joblib.dump(best_rf, 'binary_classification/best_binary_model.pkl')
    data_processed = {
        'scaler': scaler,
        'label_encoder': le
    }
    joblib.dump(data_processed, 'binary_classification/processed_data_binary.pkl')

    # Evaluation
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    with open('binary_classification/binary_evaluation_report.txt', 'w') as f:
        f.write(f"Binary Classification (Normal vs Abnormal) - Random Forest\n")
        f.write(f"Accuracy: {test_score:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n\nBest Parameters:\n")
        f.write(str(grid.best_params_))

    print("\nRandom Forest Binary model saved.")

if __name__ == "__main__":
    train_binary_rf()
