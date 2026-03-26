import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

def train_lgbm():
    print("Loading data...")
    df = pd.read_csv('data.csv')

    le = LabelEncoder()
    df['class_encoded'] = le.fit_transform(df['class'])

    X = df.drop(['class', 'class_encoded'], axis=1)
    y = df['class_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    print("\nTraining LightGBM models manually...")

    configs = [
        {'n_estimators': 50, 'learning_rate': 0.05, 'num_leaves': 15, 'min_child_samples': 5},
        {'n_estimators': 100, 'learning_rate': 0.1, 'num_leaves': 31, 'min_child_samples': 10},
        {'n_estimators': 20, 'learning_rate': 0.01, 'num_leaves': 7, 'min_child_samples': 5},
    ]

    best_lgbm = None
    best_acc = 0

    for config in configs:
        model = lgb.LGBMClassifier(**config, random_state=42, verbose=-1)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Config {config}: Accuracy = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_lgbm = model

    print(f"\nBest accuracy: {best_acc:.4f}")

    # Save files
    joblib.dump(best_lgbm, 'lightgbm_experiment/best_lgbm_model.pkl')
    data_processed = {
        'scaler': scaler,
        'label_encoder': le
    }
    joblib.dump(data_processed, 'lightgbm_experiment/processed_data_lgbm.pkl')

    y_pred = best_lgbm.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    with open('lightgbm_experiment/lgbm_evaluation_report.txt', 'w') as f:
        f.write(f"LightGBM Accuracy: {best_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nBest Config:\n")
        f.write(str(best_lgbm.get_params()))

    print("\nLightGBM results saved.")

if __name__ == "__main__":
    train_lgbm()
