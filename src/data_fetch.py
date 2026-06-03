import os
import pandas as pd
from sklearn.datasets import fetch_openml


def download_data():
    print("Pobieranie danych Palmer Penguins z OpenML...")
    # Fetch data using the ID specified in the task
    penguins = fetch_openml(data_id=42585, as_frame=True, parser='auto')
    df = penguins.frame

    # Ensure the data folder exists
    os.makedirs('data', exist_ok=True)

    # Save raw data to a CSV file
    output_path = 'data/raw_penguins.csv'
    df.to_csv(output_path, index=False)
    print(f"Pomyślnie zapisano dane do {output_path}")


if __name__ == "__main__":
    download_data()