
import pandas as pd
import re
import string
import nltk

# Télécharger les stopwords si besoin
nltk.download("stopwords")
from nltk.corpus import stopwords

# Charger les données
df = pd.read_csv("database_clean.csv")

# Liste des mois en français
mois = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre"
]

# Stopwords français et anglais
stop_words = set(stopwords.words("french") + stopwords.words("english"))

# Mots courts qu'on souhaite garder même s'ils font <= 2 lettres
important_short_words = {"bi", "ml", "ai", "it", "nlp", "llm", "rh"}

# Mots spécifiques à exclure (niveau ou type de contrat, etc.)
remove_terms = {
    "assistant", "assistante", "apprenti", "apprentie", "alternant", "alternante",
    "stagiaire", "junior", "confirmé", "confirmée", "senior", "stage", "alternance",
    "freelance", "mois", "h/f", "f/h", "hf", "paris", "lyon", "marseille", "toulouse", "lille", "nantes", "bordeaux", "strasbourg", "rennes", "nice",
        "tours", "grenoble", "rouen", "dijon", "angers", "reims", "orléans", "montpellier", "amiens", "metz", "france", "ile"
}

# Fonction de nettoyage avancé
def clean_title_advanced(text, company_name):
    text = text.lower()

    # Suppression mentions communes
    text = re.sub(r"\b(recrute|offre|emploi|poste|h/f|f/h|cdi|cdd)\b", "", text)

    # Supprimer le nom d'entreprise
    if isinstance(company_name, str) and len(company_name.strip()) > 0:
        pattern = re.escape(company_name.lower())
        text = re.sub(pattern, "", text)

    # Supprimer les mois
    for mois_nom in mois:
        text = re.sub(rf"\b{mois_nom}\b", "", text)

    # Supprimer ponctuation et chiffres
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))

    # Nettoyer les tokens
    tokens = text.split()
    tokens = [
        t for t in tokens
        if (t not in stop_words)
        and (t not in remove_terms)
        and (len(t) > 2 or t in important_short_words)
    ]

    return " ".join(tokens)

# Application de la fonction
df["cleaned_title"] = df.apply(lambda row: clean_title_advanced(row["title"], row["company"]), axis=1)

# Aperçu
print(df[["title", "company", "cleaned_title"]].head(20))

# Sauvegarde
df.to_csv("database_cleaned_titles_v3.csv", index=False)
print("✅ Fichier sauvegardé : database_cleaned_titles_v3.csv")
