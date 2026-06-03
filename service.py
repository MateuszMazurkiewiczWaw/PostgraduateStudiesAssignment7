import bentoml
import pandas as pd


# 1. Define the service as a class with the @bentoml.service decorator
@bentoml.service(name="penguins_service")
class PenguinsService:

    # 2. Initialization method - models are loaded into memory here
    def __init__(self):
        self.model = bentoml.sklearn.load_model("penguins_classifier:latest")
        self.encoder = bentoml.sklearn.load_model("penguins_encoder:latest")

    # 3. Define the endpoint using the @bentoml.api decorator
    # In the new BentoML, we don't use bentoml.io.JSON; standard Python types (dict) are sufficient
    @bentoml.api
    def predict(self, input_data: dict) -> dict:
        # Create a DataFrame from the received dictionary
        df = pd.DataFrame([input_data])

        # Separate numerical and categorical features
        num_cols = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
        cat_cols = ['island', 'sex']

        # Use our encoder to transform text into zeros and ones
        encoded_cats = self.encoder.transform(df[cat_cols])
        encoded_df = pd.DataFrame(
            encoded_cats,
            columns=self.encoder.get_feature_names_out(cat_cols)
        )

        # Combine everything into the final feature set
        final_features = pd.concat([df[num_cols], encoded_df], axis=1)

        # Make a prediction using the loaded model
        prediction = self.model.predict(final_features)

        # Return the result
        return {"predicted_species": prediction[0]}