from pathlib import Path
import pandas as pd
import pickle

__version__ = "0.1.4"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as saved_model:
    model = pickle.load(saved_model)


def predict_demand_pipeline(country_code, year):
    # Load the encoding mapping file
    with open(f"{BASE_DIR}/encoding_mapping.csv", "rb") as csv_encoding:
        encoding_mapping = pd.read_csv(csv_encoding, encoding="utf-8")

        # Load the encoding mapping file
    with open(f"{BASE_DIR}/countries_mapping.csv", "rb") as countries_encoding:
        countries_mapping = pd.read_csv(countries_encoding, encoding="utf-8")

    # Assuming 'user_country_code' is the user-provided country code
    user_country_code_encoded = encoding_mapping[encoding_mapping['Country Code'] == country_code.upper()].iloc[:, 1:]

    # Get the country name for the provided country code
    user_country_name = countries_mapping.loc[countries_mapping['Country Code'] == country_code.upper(), 'Country Name'].iloc[0]

    # Add the 'Year' column to the front of the dataframe
    user_country_code_encoded.insert(0, 'Year', int(year))

    # Use the trained model to predict the value for the user-provided country and year
    predicted_value = model.predict(user_country_code_encoded)
    adjusted_predicted_value_rmse = predicted_value * 1.2
    print(
        f'Predicted Value for {user_country_name} in {user_country_code_encoded.iloc[0]["Year"]}: {adjusted_predicted_value_rmse[0]}')
    return round(adjusted_predicted_value_rmse[0], 2)
