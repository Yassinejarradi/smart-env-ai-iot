import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# === Paths (comme dans les autres scripts) ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # remonte depuis /src
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

LABELED_CSV_PATH = os.path.join(DATA_DIR, "env_dataset_labeled.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "comfort_model.pkl")


# === Helpers ===
@st.cache_resource
def load_model():
    """
    Charge le modèle entraîné depuis le fichier .pkl
    Utilise la mise en cache de Streamlit pour éviter de le recharger à chaque interaction.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model


@st.cache_data
def load_dataset_sample(n_rows=200):
    """
    Charge un échantillon du dataset pour l'afficher dans le dashboard.
    """
    if not os.path.exists(LABELED_CSV_PATH):
        return None

    df = pd.read_csv(LABELED_CSV_PATH)
    if len(df) > n_rows:
        df = df.sample(n_rows, random_state=42).sort_index()
    return df


def predict_comfort(model, temperature, humidity):
    """
    Utilise le modèle pour prédire le confort à partir de la température et de l'humidité.
    Retourne la classe prédite (0/1) et la probabilité.
    """
    X = np.array([[temperature, humidity]])  # shape (1, 2)
    proba = model.predict_proba(X)[0]  # [p(class=0), p(class=1)]
    pred = model.predict(X)[0]         # 0 ou 1

    return int(pred), float(proba[1])  # on retourne proba d'être confortable


# === Main Streamlit app ===
def main():
    st.set_page_config(
        page_title="Smart Environment – AI + IoT Simulation",
        page_icon="🌡️",
        layout="centered",
    )

    st.title("🌡️ Smart Environment Monitoring (AI + IoT Simulation)")
    st.write(
        """
        Ce dashboard simule un système IoT qui surveille la température et l'humidité,
        et utilise un modèle de **Machine Learning** (régression logistique) pour prédire
        si l'environnement est **confortable** ou **inconfortable**.
        """
    )

    # Charger le modèle
    try:
        model = load_model()
    except FileNotFoundError as e:
        st.error(
            "Modèle introuvable. Assure-toi d'avoir exécuté `train_model.py` pour "
            "générer `models/comfort_model.pkl`."
        )
        st.stop()

    # Tabs (onglets) : un pour la prédiction en direct, un pour le dataset
    tab1, tab2 = st.tabs(["🔮 Prédiction en direct", "📊 Aperçu du dataset"])

    with tab1:
        st.subheader("🔮 Prédire le confort en fonction de la température et de l'humidité")

        # Sliders pour la température et l'humidité
        col1, col2 = st.columns(2)

        with col1:
            temperature = st.slider(
                "Température (°C)",
                min_value=10.0,
                max_value=40.0,
                value=24.0,
                step=0.5,
            )

        with col2:
            humidity = st.slider(
                "Humidité (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
            )

        # Bouton pour prédire
        if st.button("Lancer la prédiction"):
            pred, proba_comfort = predict_comfort(model, temperature, humidity)

            st.write("---")
            st.write(f"**Température entrée :** {temperature} °C")
            st.write(f"**Humidité entrée :** {humidity} %")

            if pred == 1:
                st.success(
                    f"✅ Environnement prédit comme **CONFORTABLE** "
                    f"(proba ≈ {proba_comfort*100:.1f} %)"
                )
            else:
                st.error(
                    f"⚠️ Environnement prédit comme **INCONFORTABLE** "
                    f"(proba de confort ≈ {proba_comfort*100:.1f} %)"
                )

            # Petite barre de progression visuelle
            st.write("Niveau de confort estimé :")
            st.progress(min(max(proba_comfort, 0.0), 1.0))

    with tab2:
        st.subheader("📊 Aperçu du dataset utilisé pour entraîner le modèle")

        df = load_dataset_sample()
        if df is None:
            st.warning(
                "Dataset introuvable. Assure-toi d'avoir généré les données avec "
                "`sensor_simulator.py` puis `build_dataset.py`."
            )
        else:
            st.write("Quelques lignes du dataset étiqueté :")
            st.dataframe(df.head(20))

            # Afficher quelques stats simples
            st.write("Distribution de la variable cible `comfort` :")
            st.bar_chart(df["comfort"].value_counts())


if __name__ == "__main__":
    main()
