
import pandas as pd
import unidecode
from rapidfuzz import process, fuzz

# Charger le fichier brut
df = pd.read_csv("database_categories.csv")

# === 1. Nettoyage de la colonne 'qualifications' ===
def clean_skills(text):
    if not isinstance(text, str):
        return ""
    skills = [unidecode.unidecode(s.lower().strip()) for s in text.split(",")]
    skills = [s for s in skills if s]
    return ", ".join(sorted(set(skills)))

df["cleaned_skills"] = df["qualifications"].apply(clean_skills)

# === 2. Mapping exact ===
raw_mapping = {
    "microsoft excel": "excel", "excel avance": "excel", "ms excel": "excel",
    "microsoft word": "word", "ms word": "word", "ms project": "project",
    "postgresql": "postgres", "postgresql avance": "postgres",
    "js": "javascript", "py": "python", "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn", "tensorflow": "tf", "pytorch": "torch",
    "google cloud": "gcp", "google cloud platform": "gcp", "amazon web services": "aws",
    "microsoft azure": "azure", "langage sql": "sql", "notions de sql": "sql",
    "big data": "hadoop", "apache spark": "spark", "apache kafka": "kafka",
    "analyse des donnees": "data analysis", "informatique decisionnelle": "business intelligence",
    "data visualisation": "data viz", "visualisation de donnees": "data viz"
}
skill_mapping = {unidecode.unidecode(k): v for k, v in raw_mapping.items()}

def apply_mapping(cell):
    if not isinstance(cell, str):
        return ""
    skills = [unidecode.unidecode(skill.strip()) for skill in cell.split(",")]
    standardized = [skill_mapping.get(s, s) for s in skills]
    return ", ".join(sorted(set(standardized)))

df["standard_skills"] = df["cleaned_skills"].apply(apply_mapping)

# === 3. Fuzzy matching complémentaire ===
# Créer un dictionnaire de similarité sur les termes uniques
all_skills = set()
for row in df["standard_skills"].dropna():
    for skill in row.split(","):
        skill = skill.strip().lower()
        if skill:
            all_skills.add(skill)

skills = sorted(all_skills)
fuzzy_mapping = {}
visited = set()

for skill in skills:
    if skill in visited:
        continue
    matches = process.extract(skill, skills, scorer=fuzz.token_sort_ratio, limit=10)
    group = [m[0] for m in matches if m[1] >= 90 and m[0] not in visited]
    canonical = min(group, key=len)
    for alias in group:
        fuzzy_mapping[alias] = canonical
        visited.add(alias)

def apply_fuzzy_mapping(cell):
    if not isinstance(cell, str):
        return ""
    skills = [s.strip().lower() for s in cell.split(",")]
    mapped = [fuzzy_mapping.get(s, s) for s in skills]
    return ", ".join(sorted(set(mapped)))

df["fuzzy_standard_skills"] = df["standard_skills"].apply(apply_fuzzy_mapping)

# === 4. Filtrage des compétences non pertinentes ===
excluded_skills = {
    "communication", "communication personnelle", "flexible", "creativite",
    "critique", "competences analytiques", "ensemble de competences", 
    "autonomie", "sens du contact", "esprit d’equipe", "gestion du stress",
    "polyvalence", "coaching", "relationnel", "soft skill", "direction generale",
    "curiosite", "collaboration", "projet", "organisation", "motivation",
    "aptitudes interpersonnelles", "capacité à travailler en équipe", "leadership",
    "initiative", "travail en equipe", "mise en oeuvre", "accompagnement", "ingenierie",
    "genie electrique", "genie mecanique", "competences interpersonnelles", "sens de l'organisation",
    "genie civil", "specifications", "resolution de problemes", "calculs", "depannage", "entretiens",
    "construction", "automation", "programmation", "redaction", "gestion de projet", "commercial", 
    "mise en service", "competences de coordination", "synthese"
}

def filter_skills(cell):
    if not isinstance(cell, str):
        return ""
    skills = [s.strip() for s in cell.split(",") if s.strip() and s.strip() not in excluded_skills]
    return ", ".join(sorted(set(skills)))

df["final_skills"] = df["fuzzy_standard_skills"].apply(filter_skills)

# Supprimer les colonnes intermédiaires
columns_to_drop = ["cleaned_skills", "standard_skills", "fuzzy_standard_skills"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# === Export final ===
df.to_csv("database_with_final_skills_pipeline.csv", index=False)
print("✅ Pipeline complet exécuté. Résultat dans 'database_with_final_skills_pipeline.csv'.")
