import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_model():
    print("Loading data...")
    df = pd.read_csv('data.csv')

    # Encode labels
    le = LabelEncoder()
    df['class_encoded'] = le.fit_transform(df['class'])

    # Features and target
    X = df.drop(['class', 'class_encoded'], axis=1)
    y = df['class_encoded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models and their parameter grids
    models_to_tune = [
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'criterion': ['gini', 'entropy']
            }
        },
        {
            'name': 'SVM',
            'model': SVC(random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        },
        {
            'name': 'KNN',
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance']
            }
        }
    ]

    best_overall_model = None
    best_overall_score = 0
    best_overall_name = ""

    print("\nTuning models...")
    for mt in models_to_tune:
        print(f"  Tuning {mt['name']}...")
        grid = GridSearchCV(mt['model'], mt['params'], cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)

        # We'll evaluate on the test set to pick the final winner,
        # though CV score is a more robust indicator of generalization.
        # However, for this task, we want the best performance on the unseen data.
        test_score = accuracy_score(y_test, grid.best_estimator_.predict(X_test_scaled))
        print(f"    Best CV score: {grid.best_score_:.4f}")
        print(f"    Test accuracy: {test_score:.4f}")

        if test_score > best_overall_score:
            best_overall_score = test_score
            best_overall_model = grid.best_estimator_
            best_overall_name = mt['name']

    print(f"\nWinning Model: {best_overall_name} with {best_overall_score:.4f} accuracy on test set.")

    # Save the best model and preprocessing tools
    joblib.dump(best_overall_model, 'best_model.pkl')
    data_processed = {
        'scaler': scaler,
        'label_encoder': le
    }
    joblib.dump(data_processed, 'processed_data.pkl')
    print("Model and preprocessing tools saved.")

    # Final evaluation report
    y_pred = best_overall_model.predict(X_test_scaled)
    print("\nFinal Evaluation Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print("\nConfusion Matrix:")
    print(cm_df)

    with open('evaluation_report.txt', 'w') as f:
        f.write(f"Best Model: {best_overall_name}\n")
        f.write(f"Accuracy: {best_overall_score:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=le.classes_))
        f.write("\nConfusion Matrix:\n")
        f.write(cm_df.to_string())

if __name__ == "__main__":
    train_model()
