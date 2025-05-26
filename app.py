import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="JobMatch - Trouvez votre emploi idéal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Chargement des données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("linkedin-ingenieur-database.csv")
        df = df.dropna(subset=["description"])
        
        # Ajout de salaires fictifs si non présents
        if "salary" not in df.columns:
            df["salary"] = [f"{random.randint(30, 80)}k€ - {random.randint(80, 120)}k€" for _ in range(len(df))]
            
        # Ajout de dates de publication si non présentes
        if "date_posted" not in df.columns:
            today = datetime.now()
            df["date_posted"] = [today - pd.Timedelta(days=random.randint(1, 30)) for _ in range(len(df))]
            df["date_posted"] = df["date_posted"].dt.strftime("%d/%m/%Y")
            
        return df
    except FileNotFoundError:
        # Données de démonstration si le fichier n'existe pas
        return create_demo_data()

def create_demo_data():
    data = {
        "title": [
            "Ingénieur DevOps", "Développeur Full Stack", "Data Scientist", "Ingénieur Cloud AWS", 
            "Développeur React", "Ingénieur Machine Learning", "Architecte Logiciel",
            "Développeur Backend Python", "Analyste en cybersécurité", "Chef de projet IT",
            "Développeur Frontend", "Ingénieur Big Data", "Consultant SAP", "Administrateur Système",
            "Développeur Mobile", "Ingénieur Réseau", "Analyste Business Intelligence", "Scrum Master"
        ],
        "company": [
            "TechCorp", "DataSystems", "AI Solutions", "CloudNine", "WebFrontier", "DeepLearn Tech",
            "SoftArch Inc", "BackendPro", "CyberDefense", "ProjectMasters", "FrontendStudio",
            "BigDataCorp", "SAPConsulting", "SystemAdmin Pro", "MobileFirst", "NetworkTech",
            "BIAnalytics", "AgileTeam"
        ],
        "location": [
            "Paris", "Lyon", "Marseille", "Bordeaux", "Lille", "Toulouse", "Nantes", "Strasbourg",
            "Nice", "Rennes", "Montpellier", "Grenoble", "Dijon", "Angers", "Brest", "Tours",
            "Clermont-Ferrand", "Nancy"
        ],
        "description": [
            "Expérience en Docker, Kubernetes et CI/CD. Maîtrise de Linux et AWS requise.",
            "Expertise en JavaScript, React, Node.js et bases de données SQL et NoSQL.",
            "Compétences en Python, machine learning, statistiques et visualisation de données.",
            "Certifications AWS, expérience en architecture cloud et automatisation.",
            "Maîtrise de React, Redux, TypeScript et CSS moderne.",
            "Expertise en TensorFlow, PyTorch, traitement de langage naturel et vision par ordinateur.",
            "Expérience en conception de systèmes distribués et microservices.",
            "Expertise en Python, Django, Flask et bases de données SQL.",
            "Compétences en analyse de vulnérabilités, tests de pénétration et réponse aux incidents.",
            "Certification PMP, expérience en méthodologies Agile et gestion d'équipes techniques.",
            "Développement d'interfaces utilisateur modernes avec HTML, CSS, JavaScript et frameworks frontend.",
            "Expérience en Hadoop, Spark, Kafka pour le traitement de données massives.",
            "Expertise en modules SAP FI/CO, configuration et intégration de systèmes.",
            "Administration de serveurs Linux/Windows, virtualisation et monitoring.",
            "Développement d'applications iOS et Android, React Native, Flutter.",
            "Configuration de réseaux, sécurité, pare-feu et protocoles de communication.",
            "Création de tableaux de bord, reporting et analyse de données avec Power BI.",
            "Animation d'équipes Scrum, facilitation de cérémonies Agile et coaching."
        ],
        "salary": [
            "45k€ - 65k€", "50k€ - 70k€", "55k€ - 85k€", "60k€ - 90k€", "40k€ - 65k€",
            "60k€ - 95k€", "70k€ - 110k€", "45k€ - 75k€", "55k€ - 85k€", "65k€ - 95k€",
            "42k€ - 62k€", "58k€ - 88k€", "52k€ - 72k€", "48k€ - 68k€", "46k€ - 71k€",
            "51k€ - 76k€", "54k€ - 79k€", "59k€ - 84k€"
        ],
        "date_posted": [
            "15/05/2025", "10/05/2025", "05/05/2025", "01/05/2025", "28/04/2025",
            "25/04/2025", "20/04/2025", "18/04/2025", "15/04/2025", "10/04/2025",
            "08/05/2025", "03/05/2025", "30/04/2025", "26/04/2025", "22/04/2025",
            "19/04/2025", "16/04/2025", "12/04/2025"
        ]
    }
    return pd.DataFrame(data)

df = load_data()

# Initialisation de l'état de session
if "page" not in st.session_state:
    st.session_state.page = "welcome"
if "profile_complete" not in st.session_state:
    st.session_state.profile_complete = False
if "selected_skills" not in st.session_state:
    st.session_state.selected_skills = []
if "selected_location" not in st.session_state:
    st.session_state.selected_location = ""
if "index" not in st.session_state:
    st.session_state.index = 0
if "matches" not in st.session_state:
    st.session_state.matches = []
if "min_salary" not in st.session_state:
    st.session_state.min_salary = 30
if "max_salary" not in st.session_state:
    st.session_state.max_salary = 120
if "experience_level" not in st.session_state:
    st.session_state.experience_level = "Tous niveaux"
if "filtered_jobs" not in st.session_state:
    st.session_state.filtered_jobs = None

# Liste des compétences disponibles
skills = [
    "Python", "Java", "C#", "JavaScript", "TypeScript", "Node.js", "React", "Vue.js", "Angular",
    "CSS", "HTML", "SQL", "NoSQL", "MongoDB", "PostgreSQL", "Azure", "AWS", "GCP",
    "Docker", "Kubernetes", "Linux", "Git", "CI/CD", "Agile", "Scrum", "DevOps",
    "Machine Learning", "Deep Learning", "Data Analysis", "Big Data", "Hadoop", "Spark",
    "TensorFlow", "PyTorch", "NLP", "Computer Vision", "Django", "Flask", "Spring Boot",
    "REST API", "GraphQL", "Microservices", "Cloud Architecture", "Mobile Development", 
    "iOS", "Android", "React Native", "Flutter", ".NET", "PHP", "Ruby", "Go", "Rust"
]

# Niveaux d'expérience
experience_levels = ["Tous niveaux", "Junior", "Intermédiaire", "Senior", "Lead"]

# Extraction de villes uniques à partir du dataset
@st.cache_data
def get_unique_cities():
    if 'location' in df.columns:
        cities = set()
        for loc in df['location'].dropna():
            for city in loc.split(','):
                cities.add(city.strip())
        return sorted(list(cities))
    return ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Strasbourg", "Montpellier"]

# Algorithme de matching amélioré
def get_matching_jobs(user_skills, user_location, min_salary=0, max_salary=999, experience="Tous niveaux"):
    df_filtered = df.copy()
    
    # Filtrage par ville (si spécifiée)
    if user_location:
        df_filtered = df_filtered[df_filtered['location'].str.lower().str.contains(user_location.lower(), na=False)]
    
    # Préparation pour le matching par compétences
    if not user_skills:
        df_filtered["matching_score"] = 0.5
        return df_filtered.sample(frac=1).reset_index(drop=True)

    # Utilisation de TF-IDF pour un meilleur matching
    user_input = " ".join(user_skills)
    tfidf = TfidfVectorizer(stop_words='english')
    
    if len(df_filtered) == 0:
        return pd.DataFrame()
        
    tfidf_matrix = tfidf.fit_transform(df_filtered["description"])
    user_vec = tfidf.transform([user_input])
    
    # Calcul de la similarité cosinus
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    df_filtered["matching_score"] = similarity_scores
    
    # Bonus pour les correspondances exactes de compétences
    for skill in user_skills:
        skill_lower = skill.lower()
        mask = df_filtered["description"].str.lower().str.contains(skill_lower, na=False)
        df_filtered.loc[mask, "matching_score"] += 0.1
    
    # Bonus pour la correspondance de localisation
    if user_location:
        location_mask = df_filtered['location'].str.lower().str.contains(user_location.lower(), na=False)
        df_filtered.loc[location_mask, "matching_score"] += 0.2
    
    # Normalisation des scores entre 0 et 1
    max_score = df_filtered["matching_score"].max()
    if max_score > 0:
        df_filtered["matching_score"] = df_filtered["matching_score"] / max_score
    
    return df_filtered.sort_values(by="matching_score", ascending=False).reset_index(drop=True)

# Styles CSS
st.markdown("""
<style>
.stApp {
    background-color: #f8f9fa;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}

.job-card {
    background-color: white;
    border-radius: 15px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    position: relative;
    margin-bottom: 2rem;
    border-top: 4px solid #6366F1;
}

.match-badge {
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: linear-gradient(135deg, #10B981, #34D399);
    color: white;
    font-weight: bold;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.9rem;
}

.company-name {
    color: #6366F1;
    font-weight: 600;
    font-size: 1.1rem;
}

.skill-badge {
    display: inline-block;
    background-color: #EEF2FF;
    color: #4F46E5;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

.welcome-container {
    text-align: center;
    padding: 4rem 2rem;
    max-width: 800px;
    margin: 0 auto;
}

.welcome-title {
    font-size: 3rem;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 2rem;
    font-weight: bold;
}

.welcome-subtitle {
    font-size: 1.5rem;
    color: #4B5563;
    margin-bottom: 3rem;
    line-height: 1.6;
}

.perfect-match {
    background: linear-gradient(45deg, #6366F1, #EC4899);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
    margin: 1rem 0;
}

.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #6B7280;
}

.nav-container {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 2rem;
    padding: 1rem;
    background-color: white;
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# Barre de navigation simplifiée
def render_navbar():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👤 Profil", use_container_width=True, type="primary" if st.session_state.page == "profile" else "secondary"):
            st.session_state.page = "profile"
            st.rerun()
    
    with col2:
        if st.button("🔍 Découvrir", use_container_width=True, type="primary" if st.session_state.page == "discover" else "secondary"):
            st.session_state.page = "discover"
            st.rerun()
    
    with col3:
        badge_text = f"💖 Matchs ({len(st.session_state.matches)})" if st.session_state.matches else "💖 Matchs"
        if st.button(badge_text, use_container_width=True, type="primary" if st.session_state.page == "matches" else "secondary"):
            st.session_state.page = "matches"
            st.rerun()

# Page d'accueil
if st.session_state.page == "welcome":
    st.markdown("""
    <div class="welcome-container">
        <h1 class="welcome-title">JobMatch 🚀</h1>
        <p class="welcome-subtitle">
            Trouvez l'emploi de vos rêves avec notre application de matching intelligent.<br>
            Swipez, matchez, postulez !
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Commencer maintenant", use_container_width=True, type="primary"):
            st.session_state.page = "profile"
            st.rerun()

# Page de création de profil
elif st.session_state.page == "profile":
    render_navbar()
    
    st.markdown("## 👤 Créez votre profil professionnel")
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Votre nom", value=st.session_state.get("name", ""))
            email = st.text_input("Votre email", value=st.session_state.get("email", ""))
        
        with col2:
            cities = get_unique_cities()
            location_index = 0
            if st.session_state.get("selected_location") and st.session_state.get("selected_location") in cities:
                location_index = cities.index(st.session_state.get("selected_location"))
            
            location = st.selectbox("Votre ville", cities, index=location_index)
            experience = st.selectbox("Niveau d'expérience", experience_levels, 
                                    index=experience_levels.index(st.session_state.get("experience_level", "Tous niveaux")))
        
        st.markdown("### Vos compétences")
        selected_skills = st.multiselect("Sélectionnez vos compétences", skills, 
                                       default=st.session_state.get("selected_skills", []))
        
        st.markdown("### Préférences salariales")
        col1, col2 = st.columns(2)
        with col1:
            min_salary = st.slider("Salaire minimum (k€)", 20, 100, st.session_state.get("min_salary", 30))
        with col2:
            max_salary = st.slider("Salaire maximum (k€)", min_salary, 150, 
                                 max(st.session_state.get("max_salary", 120), min_salary))
        
        if st.form_submit_button("💾 Enregistrer mon profil", use_container_width=True, type="primary"):
            if name and location and selected_skills:
                st.session_state.name = name
                st.session_state.email = email
                st.session_state.selected_location = location
                st.session_state.selected_skills = selected_skills
                st.session_state.experience_level = experience
                st.session_state.min_salary = min_salary
                st.session_state.max_salary = max_salary
                st.session_state.profile_complete = True
                
                # Filtrer les offres selon le profil
                st.session_state.filtered_jobs = get_matching_jobs(
                    selected_skills, location, min_salary, max_salary, experience
                )
                st.session_state.index = 0
                
                st.success("✅ Profil enregistré avec succès !")
                st.balloons()
                
                # Redirection automatique après 2 secondes
                import time
                time.sleep(1)
                st.session_state.page = "discover"
                st.rerun()
            else:
                st.error("❌ Veuillez remplir tous les champs obligatoires (nom, ville et compétences).")

# Page de découverte d'emplois
elif st.session_state.page == "discover":
    render_navbar()
    
    if not st.session_state.profile_complete:
        st.warning("⚠️ Veuillez d'abord compléter votre profil pour voir les offres personnalisées.")
        if st.button("Aller au profil", type="primary"):
            st.session_state.page = "profile"
            st.rerun()
    else:
        # Filtrage des offres si pas encore fait
        if st.session_state.filtered_jobs is None:
            st.session_state.filtered_jobs = get_matching_jobs(
                st.session_state.selected_skills,
                st.session_state.selected_location,
                st.session_state.min_salary,
                st.session_state.max_salary,
                st.session_state.experience_level
            )
            st.session_state.index = 0
        
        jobs = st.session_state.filtered_jobs
        total_jobs = len(jobs)
        
        if total_jobs == 0:
            st.markdown("""
            <div class="empty-state">
                <h3>😔 Aucune offre ne correspond à vos critères</h3>
                <p>Essayez de modifier vos filtres ou d'élargir votre recherche</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Modifier mon profil", type="primary"):
                st.session_state.page = "profile"
                st.rerun()
        else:
            index = st.session_state.index
            
            # Barre de progression
            progress_percent = min(index / total_jobs, 1.0)
            st.progress(progress_percent, text=f"Offre {index + 1} sur {total_jobs}")
            
            if index >= total_jobs:
                st.markdown("""
                <div class="empty-state">
                    <h3>🎉 Vous avez parcouru toutes les offres !</h3>
                    <p>Consultez vos matchs ou modifiez vos critères pour découvrir de nouvelles opportunités</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💖 Voir mes matchs", type="primary", use_container_width=True):
                        st.session_state.page = "matches"
                        st.rerun()
                with col2:
                    if st.button("⚙️ Modifier mon profil", use_container_width=True):
                        st.session_state.page = "profile"
                        st.rerun()
            else:
                # Affichage de l'offre actuelle
                job = jobs.iloc[index]
                matching_score = int(job.get("matching_score", 0.5) * 100)
                is_perfect_match = matching_score >= 90
                
                # Carte de l'offre
                st.markdown(f"""
                <div class="job-card">
                    <div class="match-badge">{matching_score}% Match</div>
                    <h2 style="margin-bottom: 0.5rem; color: #1F2937;">{job['title']}</h2>
                    <p class="company-name" style="margin-bottom: 1.5rem;">{job['company']}</p>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
                        <div>
                            <p style="color: #6B7280; margin-bottom: 0.2rem; font-weight: 500;">📍 Localisation</p>
                            <p style="color: #374151; font-weight: 600;">{job['location']}</p>
                        </div>
                        <div>
                            <p style="color: #6B7280; margin-bottom: 0.2rem; font-weight: 500;">💰 Salaire</p>
                            <p style="color: #059669; font-weight: 600;">{job.get('salary', 'Non précisé')}</p>
                        </div>
                        <div>
                            <p style="color: #6B7280; margin-bottom: 0.2rem; font-weight: 500;">📅 Publié le</p>
                            <p style="color: #374151; font-weight: 600;">{job.get('date_posted', 'Récemment')}</p>
                        </div>
                    </div>
                    
                    <h3 style="margin-bottom: 1rem; color: #1F2937;">Description du poste</h3>
                    <p style="color: #4B5563; line-height: 1.7; margin-bottom: 1.5rem;">{job['description']}</p>
                    
                    <h3 style="margin-bottom: 1rem; color: #1F2937;">Compétences requises</h3>
                    <div style="margin-bottom: 2rem;">
                """, unsafe_allow_html=True)
                
                # Extraction et affichage des compétences
                job_text = job['description'].lower()
                matched_skills = [skill for skill in skills if skill.lower() in job_text]
                
                for skill in matched_skills[:10]:
                    st.markdown(f'<span class="skill-badge">{skill}</span>', unsafe_allow_html=True)
                
                st.markdown("</div></div>", unsafe_allow_html=True)
                
                # Boutons de swipe
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("👎 Passer", use_container_width=True):
                        st.session_state.index += 1
                        st.rerun()
                
                # Bloc de feedback intermédiaire
                if st.session_state.get("feedback_step", False):
                    job_dict = st.session_state.get("pending_like_job")
                    st.success("💖 Vous avez aimé cette offre. Dites-nous pourquoi :")

                    choix = st.multiselect("Qu'avez-vous apprécié dans cette offre ?", [
                        "Le titre du poste",
                        "Les compétences requises",
                        "La localisation",
                        "Le secteur d'activité",
                        "Le salaire proposé"
                    ])

                    if st.button("✅ Valider mes préférences"):
                        feedback = {}
                        key_map = {
                            "Le titre du poste": "title",
                            "Les compétences requises": "final_skills",
                            "La localisation": "location",
                            "Le secteur d'activité": "secteur_activite",
                            "Le salaire proposé": "salary"
                        }
                        for r in choix:
                            if r in key_map:
                                feedback[key_map[r]] = "👍 J'aime"

                            # 🔐 Sécurisation des attributs de session
                        if "user_feedback" not in st.session_state:
                            st.session_state.user_feedback = []
                        if "user_preferences" not in st.session_state:
                            st.session_state.user_preferences = {}

                        st.session_state.user_feedback.append({
                            "job": job_dict,
                            "feedback": feedback
                        })
                        st.session_state.matches.append(job_dict)

                        for r in choix:
                            st.session_state.user_preferences[r] = st.session_state.user_preferences.get(r, 0) + 1

                        st.success("✅ Vos préférences ont été enregistrées !")
                        st.session_state.feedback_step = False
                        st.session_state.pending_like_job = None
                        st.session_state.index += 1
                        st.rerun()

                    st.stop()

                # Bouton J'aime qui active le feedback
                with col3:
                    if st.button("👍 J'aime", use_container_width=True, type="primary"):
                        st.session_state.pending_like_job = job.to_dict()
                        st.session_state.feedback_step = True
                        st.rerun()


# Page des matchs
elif st.session_state.page == "matches":
    render_navbar()
    
    st.markdown("## 💖 Vos offres matchées")
    
    if not st.session_state.matches:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size: 4rem; margin-bottom: 2rem;">💔</div>
            <h3>Vous n'avez pas encore de match</h3>
            <p>Swipez à droite sur les offres qui vous intéressent pour les retrouver ici</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Découvrir des offres", type="primary"):
            st.session_state.page = "discover"
            st.rerun()
    else:
        # Statistiques des matchs
        perfect_matches = len([m for m in st.session_state.matches if m.get('perfect_match', False)])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total des matchs", len(st.session_state.matches))
        with col2:
            st.metric("Matchs parfaits", perfect_matches)
        with col3:
            avg_score = np.mean([m.get('matching_score', 0.5) for m in st.session_state.matches]) * 100
            st.metric("Score moyen", f"{int(avg_score)}%")
        
        # Filtres pour les matches
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox("Trier par", ["Score de match", "Plus récent", "Entreprise"], index=0)
        with col2:
            match_filter = st.selectbox("Filtrer", ["Tous les matchs", "Matchs parfaits uniquement"], index=0)

        # Application des filtres
        matches = st.session_state.matches.copy()
        
        if match_filter == "Matchs parfaits uniquement":
            matches = [match for match in matches if match.get('perfect_match', False)]
        
        if sort_by == "Score de match":
            matches = sorted(matches, key=lambda x: x.get('matching_score', 0), reverse=True)
        elif sort_by == "Plus récent":
            matches = sorted(matches, key=lambda x: x.get('date_posted', ''), reverse=True)
        elif sort_by == "Entreprise":
            matches = sorted(matches, key=lambda x: x.get('company', '').lower())
        
        if not matches and match_filter == "Matchs parfaits uniquement":
            st.info("Vous n'avez pas encore de match parfait. Essayez d'afficher tous les matchs.")
        else:
            # Affichage des matches
            for i, match in enumerate(matches):
                with st.container():
                    if match.get('perfect_match', False):
                        st.markdown('<div class="perfect-match">✨ Match Parfait! 100% de compatibilité</div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background-color: white; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-left: 4px solid #6366F1;">
                            <h3 style="margin-bottom: 0.5rem; color: #1F2937;">{match['title']}</h3>
                            <p class="company-name" style="margin-bottom: 1rem;">{match['company']}</p>
                            
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                                <div>
                                    <p style="color: #6B7280; margin-bottom: 0.2rem; font-size: 0.9rem;">📍 Localisation</p>
                                    <p style="color: #374151; font-weight: 500;">{match.get('location', 'Non précisé')}</p>
                                </div>
                                <div>
                                    <p style="color: #6B7280; margin-bottom: 0.2rem; font-size: 0.9rem;">💰 Salaire</p>
                                    <p style="color: #059669; font-weight: 500;">{match.get('salary', 'Non précisé')}</p>
                                </div>
                                <div>
                                    <p style="color: #6B7280; margin-bottom: 0.2rem; font-size: 0.9rem;">📅 Publié le</p>
                                    <p style="color: #374151; font-weight: 500;">{match.get('date_posted', 'Récemment')}</p>
                                </div>
                            </div>
                            
                            <div style="margin-bottom: 1rem;">
                                <h4 style="color: #374151; margin-bottom: 0.5rem;">Description</h4>
                                <p style="color: #4B5563; line-height: 1.6;">{match['description'][:300]}{"..." if len(match['description']) > 300 else ""}</p>
                            </div>
                            
                            <div style="margin-bottom: 1rem;">
                                <h4 style="color: #374151; margin-bottom: 0.5rem;">Compétences requises</h4>
                        """, unsafe_allow_html=True)
                        
                        # Compétences correspondantes
                        job_text = match['description'].lower()
                        matched_skills = [skill for skill in skills if skill.lower() in job_text]
                        
                        for skill in matched_skills[:8]:
                            st.markdown(f'<span class="skill-badge">{skill}</span>', unsafe_allow_html=True)
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    with col2:
                        matching_score = int(match.get("matching_score", 0.5) * 100)
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #6366F1, #8B5CF6); 
                                    color: white; padding: 1.5rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                            <h2 style="margin: 0; font-size: 2rem;">{matching_score}%</h2>
                            <p style="margin: 0; font-size: 0.9rem;">Match</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Boutons d'action
                        if st.button("✉️ Postuler", key=f"apply_{i}", type="primary", use_container_width=True):
                            st.success("🎉 Candidature envoyée avec succès!")
                            st.balloons()
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("📋", key=f"details_{i}", help="Voir détails"):
                                with st.expander("Détails complets", expanded=True):
                                    st.write("**Description complète :**")
                                    st.write(match['description'])
                                    
                                    st.write("**Toutes les compétences détectées :**")
                                    all_skills = [skill for skill in skills if skill.lower() in match['description'].lower()]
                                    for skill in all_skills:
                                        st.markdown(f'<span class="skill-badge">{skill}</span>', unsafe_allow_html=True)
                        
                        with col_b:
                            if st.button("🗑️", key=f"remove_{i}", help="Supprimer"):
                                st.session_state.matches.remove(match)
                                st.success("Match supprimé!")
                                st.rerun()
                
                st.markdown("---")
            
            # Bouton pour retourner à la découverte
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔍 Découvrir plus d'offres", type="secondary", use_container_width=True):
                    st.session_state.page = "discover"
                    st.rerun()

# Footer avec informations supplémentaires
st.markdown("<br><br>", unsafe_allow_html=True)

# Sidebar avec informations sur l'application (optionnel)
with st.sidebar:
    if st.session_state.profile_complete:
        st.markdown(f"### 👋 Bonjour {st.session_state.get('name', 'Utilisateur')}!")
        st.markdown(f"**📍 Localisation :** {st.session_state.selected_location}")
        st.markdown(f"**🎯 Compétences :** {len(st.session_state.selected_skills)}")
        st.markdown(f"**💖 Matchs :** {len(st.session_state.matches)}")
        
        if st.button("🔄 Réinitialiser les filtres"):
            st.session_state.filtered_jobs = None
            st.session_state.index = 0
            st.success("Filtres réinitialisés!")
            st.rerun()
        
        if st.button("❌ Supprimer tous les matchs"):
            if st.session_state.matches:
                st.session_state.matches = []
                st.success("Tous les matchs ont été supprimés!")
                st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques")
    st.markdown(f"**Total d'offres :** {len(df)}")
    if st.session_state.filtered_jobs is not None:
        st.markdown(f"**Offres filtrées :** {len(st.session_state.filtered_jobs)}")
    
    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.markdown("""
    **JobMatch** utilise l'intelligence artificielle pour vous proposer 
    les offres d'emploi les plus pertinentes selon votre profil.
    
    Développé avec ❤️ en utilisant Streamlit et scikit-learn.
    """)

# Footer principal
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem; background-color: white; border-radius: 15px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
    <h3 style="color: #374151; margin-bottom: 1rem;">🚀 JobMatch - Trouvez l'emploi de vos rêves</h3>
    <p style="color: #6B7280; margin-bottom: 0;">
        Créé par Antoine, Roman & Tess<br>
        © 2025 JobMatch. Tous droits réservés.
    </p>
</div>
""", unsafe_allow_html=True)

# Script JavaScript pour améliorer l'expérience utilisateur (optionnel)
st.markdown("""
<script>
// Scroll automatique vers le haut lors du changement de page
if (window.location.hash !== window.lastHash) {
    window.scrollTo(0, 0);
    window.lastHash = window.location.hash;
}

// Animation de chargement
window.addEventListener('load', function() {
    document.body.style.opacity = '1';
    document.body.style.transition = 'opacity 0.3s ease-in-out';
});
</script>
""", unsafe_allow_html=True)

# Gestion des raccourcis clavier (optionnel)
st.markdown("""
<script>
document.addEventListener('keydown', function(event) {
    // Flèche droite ou espace pour "J'aime"
    if ((event.code === 'ArrowRight' || event.code === 'Space') && !event.target.matches('input, textarea, select')) {
        event.preventDefault();
        const likeButton = document.querySelector('button[data-testid*="baseButton-primary"]');
        if (likeButton && likeButton.textContent.includes('J\'aime')) {
            likeButton.click();
        }
    }
    
    // Flèche gauche pour "Passer"
    if (event.code === 'ArrowLeft' && !event.target.matches('input, textarea, select')) {
        event.preventDefault();
        const passButton = document.querySelector('button[data-testid*="baseButton-secondary"]');
        if (passButton && passButton.textContent.includes('Passer')) {
            passButton.click();
        }
    }
});
</script>
""", unsafe_allow_html=True)

# Message d'aide pour les raccourcis (affiché uniquement sur la page discover)
if st.session_state.page == "discover" and st.session_state.profile_complete:
    st.markdown("""
    <div style="position: fixed; bottom: 20px; right: 20px; background-color: #1F2937; color: white; 
                padding: 0.5rem 1rem; border-radius: 10px; font-size: 0.8rem; opacity: 0.8; z-index: 1000;">
        💡 Raccourcis : ← Passer | → J'aime | Espace J'aime
    </div>
    """, unsafe_allow_html=True)

# Auto-refresh pour maintenir l'application active (optionnel)
# import time
# if st.session_state.get("auto_refresh", False):
#     time.sleep(30)  # Rafraîchit toutes les 30 secondes
#     st.rerun()
