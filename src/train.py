import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn
import optuna
import joblib
import yaml
import os


def train():
    print("Rozpoczynam optymalizację hiperparametrów modelu...")

    # 1. Wczytanie parametrów z pliku params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]
    n_trials = params["n_trials"]
    random_state = params["random_state"]

    # 2. Wczytanie danych treningowych
    X_train = pd.read_csv('data/split/X_train.csv')
    y_train = pd.read_csv('data/split/y_train.csv')

    # 3. Konfiguracja MLflow (nazwa eksperymentu, pod którą będą grupowane próby)
    mlflow.set_experiment("Penguins_Classification")

    # 4. Funkcja celu dla Optuny - to tutaj odbywa się nauka
    def objective(trial):
        # Rozpoczynamy śledzenie tej konkretnej próby w MLflow
        with mlflow.start_run(nested=True):
            # Optuna losuje/wybiera hiperparametry z podanych zakresów
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 15)

            # Tworzymy model z wybranymi przez Optunę parametrami
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )

            # Trenujemy model i oceniamy go na danych treningowych (walidacja krzyżowa)
            score = cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy').mean()

            # Logujemy informacje do MLflow
            mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
            mlflow.log_metric("cv_accuracy", score)

            # Trenujemy na całości by zapisać gotowy model w tej próbie MLflow
            rf.fit(X_train, y_train)
            mlflow.sklearn.log_model(rf, "model")

            # Zwracamy wynik dla Optuny, aby wiedziała, czy poszło jej dobrze
            return score

    # 5. Uruchomienie badań Optuny (szukamy maksymalnej dokładności)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # 6. Po zakończeniu 25 prób, wyciągamy najlepsze ustawienia
    print(f"Najlepsze parametry: {study.best_params}")
    best_model = RandomForestClassifier(**study.best_params, random_state=random_state)
    best_model.fit(X_train, y_train)

    # Zapisanie ostatecznego modelu na dysk, by DVC miało go w grafie i by użyć go w ewaluacji
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/model.joblib')
    print("Zapisano najlepszy model do models/model.joblib")


if __name__ == "__main__":
    train()