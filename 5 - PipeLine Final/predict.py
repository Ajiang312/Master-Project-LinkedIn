import joblib
import pandas as pd
from utils import split_skills

# === Chargement du modèle entraîné
model = joblib.load("model_feedback.pkl")

# === Prédiction individuelle (pour debug ou tests)
def predict_like(final_skills, location, secteur_activite,
                 feedback_competence, feedback_ville, feedback_secteur):
    input_data = pd.DataFrame([{
        "final_skills": final_skills,
        "location": location,
        "secteur_activite": secteur_activite,
        "feedback_competence": int(feedback_competence),
        "feedback_ville": int(feedback_ville),
        "feedback_secteur": int(feedback_secteur)
    }])
    proba = model.predict_proba(input_data)[0][1]
    return proba

# === Prédiction batch (recommandée pour Streamlit)
def predict_batch(df):
    # Le modèle attend les colonnes suivantes :
    expected_cols = ["final_skills", "location", "secteur_activite",
                     "feedback_competence", "feedback_ville", "feedback_secteur"]
    
    # S'assurer que toutes les colonnes sont bien là
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0  # défaut = pas de feedback

    df = df[expected_cols].copy()
    df["feedback_competence"] = df["feedback_competence"].astype(int)
    df["feedback_ville"] = df["feedback_ville"].astype(int)
    df["feedback_secteur"] = df["feedback_secteur"].astype(int)

    return model.predict_proba(df)[..., 1]  # retourne les proba de LIKE (colonne 1)
