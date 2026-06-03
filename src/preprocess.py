import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
import os


def preprocess():
    print("Rozpoczynam preprocessing danych...")

    # Load raw data
    df = pd.read_csv('data/raw_penguins.csv')

    # 1. Remove missing values
    df = df.dropna()

    # 2. Extract categorical features
    categorical_cols = ['island', 'sex']

    # Initialize OneHotEncoder
    # sparse_output=False returns a regular array (easier to concatenate with pandas)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit (learn categories) and transform into zeros and ones
    encoded_array = encoder.fit_transform(df[categorical_cols])

    # Create a new DataFrame with appropriate column names
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    # Drop old categorical columns and concatenate the new encoded ones
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)

    # Save preprocessed data
    df.to_csv('data/preprocessed_penguins.csv', index=False)

    # Save the encoder itself to a file for later use in BentoML!
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoder, 'models/encoder.joblib')

    print("Zakończono! Zapisano data/preprocessed_penguins.csv oraz models/encoder.joblib")


if __name__ == "__main__":
    preprocess()