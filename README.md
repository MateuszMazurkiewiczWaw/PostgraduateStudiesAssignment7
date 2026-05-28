# Palmer Penguins ML Pipeline

Final project for the **Implementing AI Solutions** course at **SGGW (Warsaw University of Life Sciences)**.

An end-to-end, production-ready Machine Learning pipeline built to classify the species of Palmer Penguins (Adelie, Chinstrap, Gentoo). This project demonstrates the implementation of MLOps best practices, including data versioning, experiment tracking, hyperparameter optimization, and model serving.

## Features

* **Data Version Control:** Pipeline orchestration and reproducibility using **DVC**.
* **Experiment Tracking:** Logging parameters, metrics, and models using **MLflow**.
* **Hyperparameter Tuning:** Automated and intelligent optimization using **Optuna**.
* **Model Serving:** Production-ready REST API deployment using **BentoML**.

## Tech Stack

* Python 3.12
* Scikit-learn (Random Forest Classifier)
* Pandas
* DVC
* MLflow
* Optuna
* BentoML

## Project Structure

```text
├── data/                   # Raw, preprocessed, and split datasets
├── models/                 # Serialized model and encoder (.joblib files)
├── src/                    # Pipeline scripts
│   ├── data_fetch.py       # Downloads data from OpenML
│   ├── preprocess.py       # Handles missing values and One-Hot Encoding
│   ├── split_data.py       # Train/Test split
│   ├── train.py            # Optuna tuning and MLflow logging
│   └── evaluate.py         # Evaluates the best model
├── dvc.yaml                # DVC pipeline definition
├── params.yaml             # Hyperparameters configuration
├── service.py              # BentoML API service definition
└── metrics.json            # Final model evaluation metrics
```
## Getting Started
1. Reproducing the Pipeline
To run the entire machine learning pipeline from data fetching to evaluation, simply execute:

Bash
```
dvc repro
```
This command ensures that only the modified stages are re-run. To force a complete re-execution, use dvc repro --force.

2. Viewing Metrics
To quickly check the performance of the current model (Accuracy and F1 Score):

Bash
```
dvc metrics show
```

3. Exploring Experiments (MLflow)
To view the detailed history of Optuna trials, hyperparameters, and model artifacts:

Bash
```
mlflow ui
```
Then, open http://127.0.0.1:5000 in your browser.

## API Deployment (BentoML)
To serve the trained model as a REST API:

1. Save the local model artifacts to the BentoML model store:

Bash
```
python save_bento.py
```

2. Start the BentoML server:

Bash
```
bentoml serve service:PenguinsService --reload
```

3. Access the Swagger UI at http://localhost:3000 to test the endpoint visually, or use the following curl command:

Bash
```
curl -X 'POST' \
  'http://localhost:3000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input_data": {
    "culmen_length_mm": 39.1,
    "culmen_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "island": "Torgersen",
    "sex": "MALE"
  }
}'
```
Expected Response:

JSON
```
{
  "predicted_species": "Adelie"
}
```