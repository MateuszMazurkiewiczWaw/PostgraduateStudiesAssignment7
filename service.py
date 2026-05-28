import bentoml
import pandas as pd


# 1. Definiujemy serwis jako klasę z dekoratorem @bentoml.service
@bentoml.service(name="penguins_service")
class PenguinsService:

    # 2. Metoda inicjalizująca - to tutaj ładujemy nasze modele do pamięci
    def __init__(self):
        self.model = bentoml.sklearn.load_model("penguins_classifier:latest")
        self.encoder = bentoml.sklearn.load_model("penguins_encoder:latest")

    # 3. Definiujemy endpoint za pomocą dekoratora @bentoml.api
    # W nowym BentoML nie używamy już bentoml.io.JSON, wystarczą standardowe typy Pythona (dict)
    @bentoml.api
    def predict(self, input_data: dict) -> dict:
        # Tworzymy z otrzymanego słownika tabelę (DataFrame)
        df = pd.DataFrame([input_data])

        # Rozdzielamy cechy numeryczne i kategoryczne
        num_cols = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
        cat_cols = ['island', 'sex']

        # Używamy naszego encodera do zamiany tekstu na zera i jedynki
        encoded_cats = self.encoder.transform(df[cat_cols])
        encoded_df = pd.DataFrame(
            encoded_cats,
            columns=self.encoder.get_feature_names_out(cat_cols)
        )

        # Łączymy wszystko w jeden ostateczny zbiór cech
        final_features = pd.concat([df[num_cols], encoded_df], axis=1)

        # Wykonujemy predykcję za pomocą załadowanego modelu
        prediction = self.model.predict(final_features)

        # Zwracamy wynik
        return {"predicted_species": prediction[0]}