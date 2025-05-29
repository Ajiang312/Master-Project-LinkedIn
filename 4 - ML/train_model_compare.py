import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# === Nettoyage des warnings ===
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# === Fonction de tokenisation ===
def split_skills(text):
    return text.split(", ")

# === Connexion Supabase (adapter si besoin) ===
user = "postgres.gaawuilypqekxqrogemp"
password = "iIMb17fClUXLRQWN"
host = "aws-0-eu-west-3.pooler.supabase.com"
port = "6543"
database = "postgres"
url = f"postgresql://{user}:{password}@{host}:{port}/{database}"

# === Chargement des données ===
engine = create_engine(url)
df = pd.read_sql("SELECT final_skills, job_category FROM offers WHERE final_skills IS NOT NULL", engine)

# === Supprimer les classes trop petites ===
min_samples = 5
valid_classes = df["job_category"].value_counts()
valid_classes = valid_classes[valid_classes >= min_samples].index
df = df[df["job_category"].isin(valid_classes)]

# === Données d'entrée ===
X = df["final_skills"]
y = df["job_category"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Liste des modèles à comparer ===
models = {
    "logreg": LogisticRegression(max_iter=1000),
    "randomforest": RandomForestClassifier(random_state=42),
    "knn": KNeighborsClassifier(),
    "naivebayes": MultinomialNB()
}

# === Pipeline et résultats ===
results = {}

for name, model in models.items():
    print(f"\n🔍 Entraînement avec {name}...")

    pipe = Pipeline([
        ("vect", TfidfVectorizer(tokenizer=split_skills)),
        ("clf", model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    print(f"✅ Accuracy ({name}) : {acc:.4f}")
    print(f"🎯 F1-score macro ({name}) : {f1:.4f}")
    print(classification_report(y_test, preds))
    results[name] = {"accuracy": acc, "f1_macro": f1}

# === Récapitulatif final ===
print("\n📊 Comparaison finale :")
for k, v in results.items():
    print(f"{k.upper()} → Accuracy : {v['accuracy']:.4f} | F1-macro : {v['f1_macro']:.4f}")
