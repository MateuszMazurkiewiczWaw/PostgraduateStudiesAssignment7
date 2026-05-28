import bentoml
import joblib


def save_to_bento():
    print("Ładowanie plików z dysku...")
    model = joblib.load('models/model.joblib')
    encoder = joblib.load('models/encoder.joblib')

    print("Zapisywanie do magazynu BentoML...")
    # Zapisujemy model
    bentoml.sklearn.save_model('penguins_classifier', model)

    # Zapisujemy encoder
    bentoml.sklearn.save_model('penguins_encoder', encoder)

    print("Gotowe! Możesz sprawdzić listę modeli wpisując: bentoml models list")


if __name__ == "__main__":
    save_to_bento()