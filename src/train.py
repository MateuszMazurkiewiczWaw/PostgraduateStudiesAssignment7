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

    # 1. Load parameters from params.yaml file
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]
    n_trials = params["n_trials"]
    random_state = params["random_state"]

    # 2. Load training data
    X_train = pd.read_csv('data/split/X_train.csv')
    y_train = pd.read_csv('data/split/y_train.csv')

    # 3. MLflow configuration (experiment name for grouping runs)
    mlflow.set_experiment("Penguins_Classification")

    # 4. Objective function for Optuna - learning happens here
    def objective(trial):
        # Start tracking this specific run in MLflow
        with mlflow.start_run(nested=True):
            # Optuna suggests hyperparameters from given ranges
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 3, 15)

            # Create a model with parameters chosen by Optuna
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )

            # Train the model and evaluate it on training data (cross-validation)
            score = cross_val_score(rf, X_train, y_train, cv=3, scoring='accuracy').mean()

            # Log information to MLflow
            mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
            mlflow.log_metric("cv_accuracy", score)

            # Train on the entire set to save the final model in this MLflow run
            rf.fit(X_train, y_train)
            mlflow.sklearn.log_model(rf, "model")

            # Return the score for Optuna so it knows how it performed
            return score

    # 5. Start Optuna study (aiming to maximize accuracy)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # 6. After 25 trials, extract the best parameters
    print(f"Najlepsze parametry: {study.best_params}")
    best_model = RandomForestClassifier(**study.best_params, random_state=random_state)
    best_model.fit(X_train, y_train)

    # Save the final model to disk for DVC tracking and evaluation
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/model.joblib')
    print("Zapisano najlepszy model do models/model.joblib")


if __name__ == "__main__":
    train()