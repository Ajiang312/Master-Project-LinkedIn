import pandas as pd
import re

# Charger les données
df = pd.read_csv("database_clean.csv")

# Supprimer les lignes où le titre, la description ou les qualifications sont manquants
df = df.dropna(subset=["title", "description", "qualifications"])

# Remplacer toutes les autres valeurs manquantes par "Non précisé"
df = df.fillna("Non précisé")

# Fonction pour supprimer les emojis
def remove_emojis(text):
    if isinstance(text, str):
        emoji_pattern = re.compile("[" 
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002500-\U00002BEF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    return text

# Fonction pour nettoyer les espaces et retours à la ligne
def clean_text(text):
    if isinstance(text, str):
        text = remove_emojis(text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return text

# Nettoyer les colonnes
df["description"] = df["description"].apply(clean_text)
df["title"] = df["title"].apply(clean_text)

# Nettoyage avancé du titre
def normalize_title(title):
    title = title.lower()
    title = re.sub(r'\(h/?f\)|h/f|f/h|\(e\)|-', '', title)
    title = re.sub(r'alternance|stage|cdi|cdd|internship|interim|apprentissage|alternant|', '', title)
    title = re.sub(r'\d+', '', title)
    return title.strip()

df["title"] = df["title"].apply(normalize_title)

# Nettoyage des qualifications
def clean_qualifications(text):
    if isinstance(text, str):
        text = re.sub(r'\b[Aa]jouter\b', '', text)
        items = [item.strip() for item in text.split(',') if item.strip()]
        seen = set()
        cleaned = []
        for item in items:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(item)
        return ', '.join(cleaned)
    return text

df["qualifications"] = df["qualifications"].apply(clean_qualifications)

# Sauvegarde du fichier nettoyé
df.to_csv("database_clean.csv", index=False)
print("✅ Fichier nettoyé et sauvegardé : database_clean.csv")
