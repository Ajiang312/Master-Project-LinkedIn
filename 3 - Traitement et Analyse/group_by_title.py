
import pandas as pd
from rapidfuzz import process, fuzz

# Charger les données
df = pd.read_csv("database_cleaned_titles_v3.csv")

category_mapping = {
    "Data Scientist": ["data scientist", "scientifique données"],
    "Data Engineer": ["big data", "data engineer", "données", "ingénieur data"],
    "Data Analyst": ["analyste données", "data analyst"],
    "ML Engineer": ["deep learning", "machine learning", "ml", "ml engineer"],
    "LLM Engineer": ["chatgpt", "large language model", "llm", "llm engineer"],
    "AI Engineer": ["ai", "ai engineer", "ia", "intelligence artificielle"],
    "Cloud Engineer": ["cloud engineer", "ingénieur cloud"],
    "DevOps Engineer": ["devops engineer", "devopsingénieure devops", "ingénieur devops"],
    "Software Engineer": ["ingénieur logiciel", "software engineer"],
    "BI Analyst": ["analyste bi", "bi analyst", "business intelligence"],
    "Security Engineer": ["security engineer", "sûreté nucléaire", "sécurité"],
    "Cybersecurity Engineer": ["cybersecurity engineer", "cybersécurité"],
    "Network Engineer": ["ingénieur réseau", "network engineer", "radiofréquence"],
    "Test Engineer": ["fiabilité", "qa", "test", "test engineer", "validation"],
    "System Engineer": ["ingénieur système", "instrumentation", "systèmes embarqués", "system engineer"],
    "IT Architect": ["architecte", "it architect"],
    "Developer": ["developer", "developpeur", "développeur"],
    "Consultant": ["consultant", "consultante"],
    "Product Owner": ["product owner"],
    "Project Manager": ["chef projet", "project manager"],
    "Manager": ["excellence opérationnelle", "manager"],
    "Technician": ["technicien"],
    "Support Engineer": ["ingénieur support", "support engineer", "soutien logistique intégré"],
    "Data Steward": ["acculturation", "data governance", "data steward", "qualité données"],
    "RPA Engineer": ["automatisation", "robotic process automation", "rpa"],
    "Data Manager": ["data manager", "data owner"],
    "Mechanical Engineer": ["conception mécanique", "mécanique", "turbine", "alternateur"],
    "Automation Engineer": ["automatisation", "automaticien", "automaticienne", "automatisme", "ingénieur automaticien", "ingénieure automaticienne"],
    "Methods Engineer": ["industrialisation", "méthodes", "méthodes industrielles", "procédés"],
    "Continuous Improvement Engineer": ["amélioration continue"],
    "Construction Engineer": ["bâtiment", "contrôle technique construction", "expert construction"],
    "Quality Engineer": ["assurance qualité", "qualité", "qualité fournisseur", "qualité fournisseurs"],
    "Full Stack Engineer": ["full stack", "ingénieur full stack", "ingénieure full stack"],
    "Installation Engineer": ["installation générale"],
    "Sales Engineer": ["business manager", "ingénieur commercial", "ingénieur d'affaires"],
    "Industrial IT Engineer": ["informatique industrielle", "production informatique"],
    "Maintenance Engineer": ["méthodes maintenance"],
    "Electronics Engineer": ["conception électronique", "électronique", "électronique puissance"],
    "Electrical Engineer": ["éclairage électricité", "électricien", "étude génie électrique", "études électricité", "électricité industrielle", "génie électrique", "conception électrique", "electrique"],
    "Functional Safety Engineer": ["sûreté fonctionnement"],
    "Validation Engineer": ["qualification validation"],
    "Java Software Engineer": ["logiciel java"],
    "Production Engineer": ["production"],
    "Graduate Site Engineer": ["graduate program", "ingénieur travaux débutant"],
    "Network Security Engineer": ["linux réseau sécurité", "réseau systèmes", "sécurité réseau"],
    "Planning Engineer": ["planification"],
    "Project Engineer": ["projets", "projets nucléaire", "projets stabilisation berges", "projets structure"],
    "Linux System Engineer": ["système linux"],
    "Energy Efficiency Engineer": ["efficacité énergétique", "performance énergétique"],
    "Industrialization Engineer": ["industrialisation", "industrialisation électronique"],
    "Environmental Engineer": ["environnement", "environnementale", "icpe", "risques industriels"],
    "Control Protection Engineer": ["commande protection"],
    "Process Engineer": ["procédé", "processus"],
    "Coastal Engineering Specialist": ["génie côtier", "stabilisation berges"],
    "Technical Sales Engineer": ["solutions systèmes sécurité", "technicocommercial", "technicocommerciale"],
    "R&D Engineer": ["recherche développement"],
    "Senior Transmission Engineer": ["sénior lignes transmission"],
    "Senior Mechanical Engineer": ["sénior mécanique industrielle"],
    "Robotics Engineer": ["automates mobiles", "robotique", "robots"],
    "Embedded Systems Engineer": ["firmware", "microcontrôleurs", "systèmes embarqués", "systèmes intégrés"],
    "SCADA Engineer": ["automatismes industriels", "scada", "supervision industrielle"],
    "Blockchain Engineer": ["blockchain", "ledger", "smart contract"],
    "Quantum Computing Engineer": ["quantique", "quantum"],
    "Telecommunications Engineer": ["antennes", "radio", "télécommunications"],
    "Geotechnical Engineer": ["fondations", "géotechnique", "terrains"],
    "Naval Engineer": ["naval", "propulsion marine", "stabilisation maritime"],
    "Railway Systems Engineer": ["ferroviaire", "signalisation", "voie ferrée"],
    "Aerospace Engineer": ["aéronautique", "mécanique des fluides", "propulsion avion"],
    "Nuclear Engineer": ["combustible", "nucléaire", "sûreté nucléaire"],
    "Civil Engineer": ["génie civil", "infrastructures", "ouvrages"],
    "Materials Engineer": ["composites", "métaux", "matériaux", "polymères"],
    "Acoustics Engineer": ["acoustique", "bruit", "vibrations"],
    "Biomedical Engineer": ["biomédical", "capteurs médicaux", "imagerie médicale"],
    "Reliability Engineer": ["fiabilité", "mtbf", "rams"],
    "Compliance Engineer": ["ce", "conformité", "reach", "réglementation"],
    "Energy Engineer": ["énergies renouvelables", "réseaux énergétiques", "thermique"],
    "BIM Engineer": ["bim", "building information modeling", "maquette numérique"],
    "ASIC Verification Engineer": ["vérification asic"],
    "IT Consultant": ["audit itconseil"],
    "Hydropower Engineer": ["hydroélectrique", "alternateur", "turbine"],
    "Urban Infrastructure Engineer": ["surveillance infrastructures urbaines", "bureau infrastructures"],
    "Structural Engineer": ["structures", "inspection ponts", "génie civil"],
    "System Integration Engineer": ["systèmes intégrés", "ingénieur systèmes intégrés"],
}

# Inversion
search_list = []
reverse_mapping = {}
for cat, variants in category_mapping.items():
    for variant in variants:
        search_list.append(variant)
        reverse_mapping[variant] = cat

# Fuzzy match principal (threshold = 50)
def fuzzy_match_primary(text, threshold=80):
    if not isinstance(text, str) or not text.strip():
        return "Autre"
    best_match, score, _ = process.extractOne(text, search_list, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        return reverse_mapping[best_match]
    # Fallback simples
    if "llm" in text and "ml" in text:
        return "ML Engineer"
    if "rpa" in text:
        return "RPA Engineer"
    if "data quality" in text or "owner" in text:
        return "Data Steward"
    return "Autre"

# Fuzzy match secondaire (threshold = 40)
def fuzzy_match_secondary(text, threshold=40):
    if not isinstance(text, str) or not text.strip():
        return "Autre"
    best_match, score, _ = process.extractOne(text, search_list, scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        return reverse_mapping[best_match]
    return "Autre"

# Première passe
df["job_category"] = df["cleaned_title"].apply(fuzzy_match_primary)

# Deuxième passe sur les "Autre"
mask_autre = df["job_category"] == "Autre"
df.loc[mask_autre, "job_category"] = df.loc[mask_autre, "cleaned_title"].apply(fuzzy_match_secondary)

# Statistiques
total = len(df)
autres = (df["job_category"] == "Autre").sum()
pourcentage = autres / total * 100

print(f"🔍 Analyse post-matching (2 passes) :")
print(f"• Total lignes         : {total}")
print(f"• Nombre 'Autre'       : {autres}")
print(f"• Pourcentage 'Autre'  : {pourcentage:.2f}%")

# Top 30 titres les plus fréquents dans "Autre"
print("\n📌 Top 30 intitulés encore classés comme 'Autre' :")
print(df[df["job_category"] == "Autre"]["cleaned_title"].value_counts().head(30))

# Export final
df.to_csv("database_categories.csv", index=False)
print("✅ Fichier final : database_categories.csv")
