import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
import os


def preprocess():
    print("Rozpoczynam preprocessing danych...")

    # Wczytanie surowych danych
    df = pd.read_csv('data/raw_penguins.csv')

    # 1. Usunięcie braków danych
    df = df.dropna()

    # 2. Wyodrębnienie cech kategorycznych
    categorical_cols = ['island', 'sex']

    # Inicjalizacja OneHotEncodera
    # sparse_output=False sprawia, że otrzymujemy zwykłą tablicę (łatwiejszą do połączenia z pandas)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Dopasowanie (nauka kategorii) i transformacja na zera i jedynki
    encoded_array = encoder.fit_transform(df[categorical_cols])

    # Utworzenie z tego nowej tabeli (DataFrame) z odpowiednimi nazwami kolumn
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    # Wyrzucenie starych kolumn tekstowych i doklejenie nowych, zakodowanych
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)

    # Zapisanie przetworzonych danych
    df.to_csv('data/preprocessed_penguins.csv', index=False)

    # Zapisanie samego encodera do pliku, by użyć go później w BentoML!
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoder, 'models/encoder.joblib')

    print("Zakończono! Zapisano data/preprocessed_penguins.csv oraz models/encoder.joblib")


if __name__ == "__main__":
    preprocess()