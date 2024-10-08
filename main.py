# main.py

from data_processing import load_data, select_columns, analyze_data
from model_training import split_data, train_random_forest
from predictions import encode_area, create_prediction_dataframe
from visualization import display_styled_dataframe
from sklearn.preprocessing import LabelEncoder

def main():
    # Charger et traiter les données
    filepath = "../Crime_Data_from_2020_to_Present.csv"
    df = load_data(filepath)
    df = select_columns(df)

    # Analyse rapide
    analyze_data(df)

    # Séparer les données et entraîner le modèle
    X_train, X_test, y_train, y_test = split_data(df)
    rf_model = train_random_forest(X_train, y_train)

    # Encoder les zones
    le = LabelEncoder()
    df = encode_area(df, le)

    # Dictionnaire des noms des zones
    zone_name_map = dict(zip(df['AREA'], df['AREA NAME']))

    # Prédictions sur des heures données
    heures = [8, 12, 17, 22, 6]
    df_predictions = create_prediction_dataframe(rf_model, le, zone_name_map, heures)

    # Afficher le tableau stylisé
    display_styled_dataframe(df_predictions)

if __name__ == "__main__":
    main()
