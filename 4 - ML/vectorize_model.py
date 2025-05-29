import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# === Fonction de tokenisation picklable ===
def split_skills(text):
    return text.split(", ")

# === Connexion Supabase ===
user = "postgres.gaawuilypqekxqrogemp"
password = "iIMb17fClUXLRQWN"
host = "aws-0-eu-west-3.pooler.supabase.com"
port = "6543"
database = "postgres"
url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

# === Chargement des données ===
engine = create_engine(url)
df = pd.read_sql("SELECT final_skills, job_category FROM offers WHERE final_skills IS NOT NULL", engine)

# === Supprimer les classes trop rares ===
min_samples = 5
valid_classes = df["job_category"].value_counts()
valid_classes = valid_classes[valid_classes >= min_samples].index
df = df[df["job_category"].isin(valid_classes)]

# === Préparation des données ===
X = df["final_skills"]
y = df["job_category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Pipelines Count et TF-IDF ===
pipelines = {
    "count": Pipeline([
        ("vect", CountVectorizer(tokenizer=split_skills)),
        ("clf", RandomForestClassifier(random_state=42))
    ]),
    "tfidf": Pipeline([
        ("vect", TfidfVectorizer(tokenizer=split_skills)),
        ("clf", RandomForestClassifier(random_state=42))
    ])
}

# === Entraînement et comparaison ===
results = {}

for name, pipe in pipelines.items():
    print(f"\n🔍 Entraînement avec {name}...")
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"✅ Accuracy ({name}) : {acc:.4f}")
    print(f"🎯 F1-score macro ({name}) : {f1:.4f}")
    print(classification_report(y_test, preds))



