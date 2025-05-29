import pandas as pd
import re

# Charger le fichier CSV
df = pd.read_csv("database_with_final_skills_pipeline.csv")

# Supprimer les lignes sans description
df = df.dropna(subset=["description"])
# Supprimer les lignes sans compétences
df = df.dropna(subset=["final_skills"])
df = df[df["final_skills"].str.strip().astype(bool)]

# Définir des mots-clés associés à chaque secteur
sector_keywords = {
    "Informatique / IT": ["développement", "software", "informatique", "devops", "cloud", "python", "java", "backend", "frontend", "fullstack"],
    "Télécommunications": ["réseau", "telecom", "5g", "fibre", "commutation", "antenne"],
    "Énergie": ["énergie", "nucléaire", "pétrole", "gaz", "éolien", "solaire", "barrage"],
    "Construction / BTP": ["chantier", "génie civil", "btp", "structure", "bâtiment", "voirie"],
    "Aéronautique / Défense": ["aéronautique", "défense", "militaire", "spatial", "radar", "drone", "missile"],
    "Automobile / Transports": ["automobile", "véhicule", "voiture", "mécanique", "transports", "ferroviaire"],
    "Finance / Banque / Assurance": ["banque", "assurance", "finance", "audit", "comptable", "investissement"],
    "Santé / Médical / Pharmaceutique": ["médical", "santé", "pharma", "biotech", "hôpital", "clinique"],
    "Agroalimentaire": ["agroalimentaire", "agriculture", "agro", "production alimentaire", "industries alimentaires"],
    "Enseignement / Recherche": ["recherche", "doctorant", "professeur", "université", "enseignement"],
    "Logistique / Supply Chain": ["logistique", "supply chain", "entrepôt", "transport", "chaîne d'approvisionnement"],
    "Conseil / Audit": ["conseil", "consultant", "audit", "cabinet", "stratégie"]
}

# Fonction d'association secteur à partir de la description
def identify_sector(description):
    if isinstance(description, str):
        desc = description.lower()
        for sector, keywords in sector_keywords.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", desc):
                    return sector
    return "Autre"

# Appliquer la détection du secteur à chaque ligne
df["secteur_activite"] = df["description"].apply(identify_sector)

# Sauvegarder dans un nouveau fichier CSV
df.to_csv("database_with_sectors.csv", index=False)
print("✅ Fichier enrichi sauvegardé : database_with_sectors.csv")
