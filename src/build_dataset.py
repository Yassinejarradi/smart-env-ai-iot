import os
import pandas as pd

# Base directories (same logique que sensor_simulator.py)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # remonte d'un niveau depuis /src
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

RAW_CSV_PATH = os.path.join(DATA_DIR, "env_dataset.csv")
LABELED_CSV_PATH = os.path.join(DATA_DIR, "env_dataset_labeled.csv")


def load_raw_data():
    """
    Charge le fichier CSV brut généré par le simulateur.
    """
    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(f"Fichier brut introuvable : {RAW_CSV_PATH}")

    df = pd.read_csv(RAW_CSV_PATH)
    print(f"Dataset brut chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def label_comfort(df):
    """
    Ajoute une colonne 'comfort' (1 = confortable, 0 = pas confortable)
    en fonction de règles simples sur la température et l'humidité.
    """

    # Règles de confort (tu peux les ajuster plus tard)
    # Ici on considère confortable si :
    # 21°C <= temp <= 26°C ET 40% <= humidity <= 60%
    conditions_comfort = (
        (df["temperature"] >= 21.0)
        & (df["temperature"] <= 26.0)
        & (df["humidity"] >= 40.0)
        & (df["humidity"] <= 60.0)
    )

    df["comfort"] = conditions_comfort.astype(int)  # True -> 1, False -> 0

    print("Colonne 'comfort' ajoutée.")
    print(df["comfort"].value_counts())
    return df


def save_labeled_data(df):
    """
    Sauvegarde le dataset étiqueté dans un nouveau fichier CSV.
    """
    df.to_csv(LABELED_CSV_PATH, index=False)
    print(f"Dataset étiqueté sauvegardé dans : {LABELED_CSV_PATH}")


def main():
    df_raw = load_raw_data()
    df_labeled = label_comfort(df_raw)
    save_labeled_data(df_labeled)


if __name__ == "__main__":
    main()
