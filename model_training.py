# model_training.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def split_data(df):
    """Sépare les données en variables X et y, puis les divise en ensembles d'entraînement et de test."""
    X = df[['TIME OCC']]
    y = df['AREA']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_random_forest(X_train, y_train):
    """Entraîne un modèle RandomForest et le retourne."""
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10)
    rf_model.fit(X_train, y_train)
    return rf_model
