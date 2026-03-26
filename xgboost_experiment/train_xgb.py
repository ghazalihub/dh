import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_xgb():
    print("Loading data...")
    df = pd.read_csv('data.csv')

    le = LabelEncoder()
    df['class_encoded'] = le.fit_transform(df['class'])

    X = df.drop(['class', 'class_encoded'], axis=1)
    y = df['class_encoded']

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    print("\nTuning XGBoost...")
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    print(f"  Best CV score: {grid.best_score_:.4f}")
    best_xgb = grid.best_estimator_

    y_pred = best_xgb.predict(X_test_scaled)
    test_score = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {test_score:.4f}")

    # Save files
    joblib.dump(best_xgb, 'xgboost_experiment/best_xgb_model.pkl')
    data_processed = {
        'scaler': scaler,
        'label_encoder': le
    }
    joblib.dump(data_processed, 'xgboost_experiment/processed_data_xgb.pkl')

    # Evaluation
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    with open('xgboost_experiment/xgb_evaluation_report.txt', 'w') as f:
        f.write(f"XGBoost Accuracy: {test_score:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n\nBest Parameters:\n")
        f.write(str(grid.best_params_))

    print("\nXGBoost results saved in 'xgboost_experiment/'.")

if __name__ == "__main__":
    train_xgb()
