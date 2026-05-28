import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_data():
    print("Rozpoczynam podział danych na zbiór treningowy i testowy...")

    # Wczytanie przetworzonych danych z poprzedniego etapu
    df = pd.read_csv('data/preprocessed_penguins.csv')

    # Oddzielenie cech (X) od naszej etykiety, którą chcemy zgadywać (y)
    # W naszym przypadku chcemy klasyfikować gatunek (species)
    X = df.drop(columns=['species'])
    y = df['species']

    # Podział danych - 20% idzie do testów, 80% do treningu
    # random_state zapewnia powtarzalność wyników przy każdym uruchomieniu potoku
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Upewnienie się, że folder docelowy istnieje (zapiszemy to w nowym folderze dla porządku)
    os.makedirs('data/split', exist_ok=True)

    # Zapisanie wynikowych zbiorów do plików CSV
    X_train.to_csv('data/split/X_train.csv', index=False)
    X_test.to_csv('data/split/X_test.csv', index=False)
    y_train.to_csv('data/split/y_train.csv', index=False)
    y_test.to_csv('data/split/y_test.csv', index=False)

    print("Zakończono podział! Pliki zapisano w katalogu data/split/")


if __name__ == "__main__":
    split_data()