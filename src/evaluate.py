import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score


def evaluate():
    print("Rozpoczynam ewaluację modelu na danych testowych...")

    # 1. Wczytanie danych testowych
    X_test = pd.read_csv('data/split/X_test.csv')
    y_test = pd.read_csv('data/split/y_test.csv')

    # 2. Wczytanie najlepszego modelu zapisanego w poprzednim etapie
    model = joblib.load('models/model.joblib')

    # 3. Wygenerowanie przewidywań (model zgaduje gatunki na podstawie X_test)
    predictions = model.predict(X_test)

    # 4. Obliczenie metryk jakości
    # f1_score z average='weighted' jest idealne dla problemów z więcej niż 2 klasami (mamy 3 gatunki)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')

    # 5. Zapisanie metryk do słownika
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1
    }

    # 6. Zapisanie do pliku metrics.json
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Ewaluacja zakończona! Wyniki: {metrics}")


if __name__ == "__main__":
    evaluate()