# data_processing.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath):
    """Charge le fichier CSV en un DataFrame pandas."""
    return pd.read_csv(filepath)

def select_columns(df):
    """Sélectionne les colonnes TIME OCC, AREA, et AREA NAME."""
    return df[['TIME OCC', 'AREA', 'AREA NAME']]

def analyze_data(df):
    """Génère des boxplots et une heatmap des variables numériques."""
    num_vars = ['TIME OCC', 'AREA']
    
    # Boxplots
    for col in num_vars:
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot de la variable "{col}"')
        plt.show()
    
    # Heatmap de corrélation
    df_num = df[num_vars]
    sns.heatmap(df_num.corr(), cmap='coolwarm', annot=True)
    plt.title('Heatmap de corrélation des variables numériques')
    plt.show()
