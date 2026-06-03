import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_data():
    print("Rozpoczynam podział danych na zbiór treningowy i testowy...")

    # Load preprocessed data from the previous stage
    df = pd.read_csv('data/preprocessed_penguins.csv')

    # Separate features (X) from the label we want to predict (y)
    # In our case, we want to classify the species
    X = df.drop(columns=['species'])
    y = df['species']

    # Split data - 20% for testing, 80% for training
    # random_state ensures reproducibility across pipeline runs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure target directory exists (saving in a new folder for better organization)
    os.makedirs('data/split', exist_ok=True)

    # Save resulting datasets to CSV files
    X_train.to_csv('data/split/X_train.csv', index=False)
    X_test.to_csv('data/split/X_test.csv', index=False)
    y_train.to_csv('data/split/y_train.csv', index=False)
    y_test.to_csv('data/split/y_test.csv', index=False)

    print("Zakończono podział! Pliki zapisano w katalogu data/split/")


if __name__ == "__main__":
    split_data()