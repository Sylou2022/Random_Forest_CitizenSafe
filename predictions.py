# predictions.py

import numpy as np
import pandas as pd

def encode_area(df, label_encoder):
    """Encode la colonne AREA et ajoute une nouvelle colonne AREA_ENCODED."""
    df['AREA_ENCODED'] = label_encoder.fit_transform(df['AREA'])
    return df

def predict_top_zones(rf_model, hour, minute, top_n=5):
    """Prédit les zones les plus exposées pour une heure donnée."""
    time_minutes = hour * 60 + minute
    probabilities = rf_model.predict_proba([[time_minutes]])[0]
    sorted_indices = np.argsort(probabilities)[::-1]
    
    top_zones = sorted_indices[:top_n]
    top_probs = probabilities[sorted_indices[:top_n]]
    
    return top_zones, top_probs

def create_prediction_dataframe(rf_model, label_encoder, zone_name_map, hours):
    """Crée un DataFrame de prédictions pour une liste d'heures données."""
    results = []
    for hour in hours:
        top_zones, top_probs = predict_top_zones(rf_model, hour, 0)
        row = {'Heure': f'{hour:02d}:00'}
        for i, (zone, prob) in enumerate(zip(top_zones, top_probs), 1):
            zone_name = label_encoder.inverse_transform([zone])[0]
            row[f'Zone {i}'] = zone_name_map[zone_name]
            row[f'Probabilité {i}'] = f'{prob:.2f}'
        results.append(row)
    return pd.DataFrame(results)
