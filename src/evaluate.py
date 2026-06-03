import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score


def evaluate():
    print("Rozpoczynam ewaluację modelu na danych testowych...")

    # 1. Load test data
    X_test = pd.read_csv('data/split/X_test.csv')
    y_test = pd.read_csv('data/split/y_test.csv')

    # 2. Load the best model saved in the previous stage
    model = joblib.load('models/model.joblib')

    # 3. Generate predictions (model guesses species based on X_test)
    predictions = model.predict(X_test)

    # 4. Calculate quality metrics
    # f1_score with average='weighted' is ideal for multi-class problems (we have 3 species)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')

    # 5. Save metrics to a dictionary
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1
    }

    # 6. Save to metrics.json file
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Ewaluacja zakończona! Wyniki: {metrics}")


if __name__ == "__main__":
    evaluate()