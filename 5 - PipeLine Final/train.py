import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import lightgbm as lgb
import joblib
from utils import split_skills

# Connexion Supabase
user = "postgres.gaawuilypqekxqrogemp"
password = "iIMb17fClUXLRQWN"
host = "aws-0-eu-west-3.pooler.supabase.com"
port = "6543"
database = "postgres"
url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
engine = create_engine(url)

# Chargement des données
interactions = pd.read_sql("SELECT * FROM interactions", engine)
offers = pd.read_sql("SELECT job_id, final_skills, location, secteur_activite FROM offers", engine)

df = interactions.merge(offers, left_on="offer_id", right_on="job_id", how="inner")
if df.empty:
    raise ValueError("❌ La table interactions est vide.")

# Cibles et features
y = df["liked"]
X = df[["final_skills", "location", "secteur_activite", 
        "feedback_competence", "feedback_ville", "feedback_secteur"]]

# Pipeline LightGBM
preprocessor = ColumnTransformer(transformers=[
    ("tfidf", TfidfVectorizer(tokenizer=split_skills, max_features=300), "final_skills"),
    ("location", OneHotEncoder(handle_unknown='ignore'), ["location"]),
    ("secteur", OneHotEncoder(handle_unknown='ignore'), ["secteur_activite"])
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("clf", lgb.LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

# Split & fit
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print("🚀 Entraînement du modèle LightGBM...")
pipeline.fit(X_train, y_train)

# Évaluation
preds = pipeline.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
print(f"✅ Accuracy : {acc:.4f}")
print(f"🎯 F1-score : {f1:.4f}")
print(classification_report(y_test, preds))

# Sauvegarde du modèle
joblib.dump(pipeline, "model_feedback.pkl")
print("💾 Modèle LightGBM sauvegardé sous model_feedback.pkl")
