import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Définition des chemins
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # remonte depuis /src
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

LABELED_CSV_PATH = os.path.join(DATA_DIR, "env_dataset_labeled.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "comfort_model.pkl")


def load_dataset():
    """
    Charge le dataset étiqueté depuis le CSV.
    """
    if not os.path.exists(LABELED_CSV_PATH):
        raise FileNotFoundError(f"Dataset étiqueté introuvable : {LABELED_CSV_PATH}")

    df = pd.read_csv(LABELED_CSV_PATH)
    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    return df


def prepare_data(df):
    """
    Sépare les features (X) et la target (y).
    Ici, X = temperature + humidity, y = comfort.
    """
    X = df[["temperature", "humidity"]].values  # 2 features
    y = df["comfort"].values                    # labels 0/1

    return X, y


def train_and_evaluate(X, y):
    """
    Divise les données en train/test, entraîne une régression logistique,
    évalue le modèle, et retourne le pipeline entraîné.
    """

    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # garde la même proportion de 0/1 dans train et test
    )

    print(f"Train set : {X_train.shape[0]} échantillons")
    print(f"Test set  : {X_test.shape[0]} échantillons")

    # Pipeline = StandardScaler (mise à l'échelle) + LogisticRegression
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression())
        ]
    )

    # Entraînement
    pipeline.fit(X_train, y_train)
    print("Modèle entraîné.")

    # Prédictions sur le test set
    y_pred = pipeline.predict(X_test)

    # Accuracy globale
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy sur le test set : {acc:.3f}")

    # Rapport plus détaillé
    print("\nClassification report :")
    print(classification_report(y_test, y_pred))

    return pipeline


def save_model(model):
    """
    Sauvegarde le modèle entraîné dans le dossier models/.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modèle sauvegardé dans : {MODEL_PATH}")


def main():
    df = load_dataset()
    X, y = prepare_data(df)
    model = train_and_evaluate(X, y)
    save_model(model)


if __name__ == "__main__":
    main()
