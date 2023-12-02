from pathlib import Path
import pandas as pd
import pickle

__version__ = "0.1.2"

BASE_DIR = Path(__file__).resolve(strict=True).parent

def load_model(modelname):
    with open(modelname, "rb") as saved_model:
        return pickle.load(saved_model)

def predict_demand_pipeline(country_code, year):
    modelname = f"{BASE_DIR}/linear_electric_usage-{__version__}.pkl"
    linear_model = load_model(modelname)

    # Load the encoding mapping file
    with open(f"{BASE_DIR}/encoding_mapping.csv", "rb") as csv_encoding:
        encoding_mapping = pd.read_csv(csv_encoding, encoding="utf-8")

        # Load the encoding mapping file
    with open(f"{BASE_DIR}/countries_mapping.csv", "rb") as countries_encoding:
        countries_mapping = pd.read_csv(countries_encoding, encoding="utf-8")

    # Assuming 'user_country_code' is the user-provided country code
    country_code_encoded = encoding_mapping[encoding_mapping['Country Code'] == country_code.upper()].iloc[:, 1:]

    # Get the country name for the provided country code
    country_name = countries_mapping.loc[countries_mapping['Country Code'] == country_code.upper(), 'Country Name'].iloc[0]

    # Add the 'Year' column to the front of the dataframe
    country_code_encoded.insert(0, 'Year', int(year))

    # Use the trained model to predict the value for the user-provided country and year
    predicted_value = linear_model.predict(country_code_encoded)
    adjusted_predicted_value_rmse = predicted_value * 1.2
    print(f'Predicted Energy Usage for {country_name} in {year}: {round(adjusted_predicted_value_rmse[0], 2)}')
    return round(adjusted_predicted_value_rmse[0], 2)

def predict_gdp_growth(country_code, year):
    modelname = f"{BASE_DIR}/xgb_gdp_growth-{__version__}.pkl"
    xgb_model = load_model(modelname)
    # Load the encoding mapping file
    with open(f"{BASE_DIR}/encoding_mapping.csv", "rb") as csv_encoding:
        encoding_mapping = pd.read_csv(csv_encoding, encoding="utf-8")

        # Load the encoding mapping file
    with open(f"{BASE_DIR}/countries_mapping.csv", "rb") as countries_encoding:
        countries_mapping = pd.read_csv(countries_encoding, encoding="utf-8")

    # Assuming 'user_country_code' is the user-provided country code
    country_code_encoded = encoding_mapping[encoding_mapping['Country Code'] == country_code.upper()].iloc[:, 1:]

    # Get the country name for the provided country code
    country_name = countries_mapping.loc[countries_mapping['Country Code'] == country_code.upper(), 'Country Name'].iloc[0]

    # Add the 'Year' column to the front of the dataframe
    country_code_encoded.insert(0, 'Year', int(year))

    # Use the trained model to predict the value for the user-provided country and year
    predicted_value = xgb_model.predict(country_code_encoded)

    print(f'Predicted GDP Growth for {country_name} in {year}: {round(predicted_value[0], 2)}')
    return round(predicted_value[0], 2)

def predict_population(country_code, year):
    modelname = f"{BASE_DIR}/xgb_population-{__version__}.pkl"
    xgb_model = load_model(modelname)
    # Load the encoding mapping file
    with open(f"{BASE_DIR}/encoding_mapping.csv", "rb") as csv_encoding:
        encoding_mapping = pd.read_csv(csv_encoding, encoding="utf-8")

        # Load the encoding mapping file
    with open(f"{BASE_DIR}/countries_mapping.csv", "rb") as countries_encoding:
        countries_mapping = pd.read_csv(countries_encoding, encoding="utf-8")

    # Assuming 'user_country_code' is the user-provided country code
    country_code_encoded = encoding_mapping[encoding_mapping['Country Code'] == country_code.upper()].iloc[:, 1:]

    # Get the country name for the provided country code
    country_name = countries_mapping.loc[countries_mapping['Country Code'] == country_code.upper(), 'Country Name'].iloc[0]

    # Add the 'Year' column to the front of the dataframe
    country_code_encoded.insert(0, 'Year', int(year))

    # Use the trained model to predict the value for the user-provided country and year
    predicted_value = xgb_model.predict(country_code_encoded)

    print(f'Predicted Population for {country_name} in {year}: {round(predicted_value[0])}')
    return round(predicted_value[0])