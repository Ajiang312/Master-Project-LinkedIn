import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from datetime import datetime
from sqlalchemy import create_engine, text
import pandas as pd
import json


# ============================== Configuration de la page ===================================================
st.set_page_config(
    page_title="JobMatch - Trouvez votre emploi idéal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================== Chargement des données ==================================================
# Connexion à Supabase
user = "postgres.gaawuilypqekxqrogemp"
password = "iIMb17fClUXLRQWN"
host = "aws-0-eu-west-3.pooler.supabase.com"
port = "6543"
database = "postgres"
url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
engine = create_engine(url)

@st.cache_data
def load_data():
    try:
        df = pd.read_sql("SELECT * FROM offers", engine)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des offres : {e}")
        return None

# ==================================== Pioche des offres aléatoires =====================================
def get_random_offer(engine):
    query = "SELECT * FROM offers ORDER BY RANDOM() LIMIT 1"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df.iloc[0].to_dict()

# =========================== Initialisation de l'état de session =================================================
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
if "current_job" not in st.session_state:
    st.session_state.current_job = get_random_offer(engine)

# =============================== Liste des compétences disponibles ==============================================
@st.cache_data
def get_unique_skills(df):
    if "final_skills" in df.columns:
        skill_set = set()
        for row in df["final_skills"].dropna():
            for skill in row.split(","):
                skill_set.add(skill.strip())
        return sorted(list(skill_set))
    return []

# =================================== Extraction de secteurs uniques à partir de la Base ===============================================

@st.cache_data
def get_unique_sectors(df):
    if "secteur_activite" in df.columns:
        sectors = set()
        for val in df["secteur_activite"].dropna():
            sectors.add(val.strip())
        return sorted(list(sectors))
    return []


# =================================== Extraction de villes uniques à partir de la Base ===============================================
@st.cache_data
def get_unique_cities(df):
    if 'location' in df.columns:
        cities = set()
        for loc in df['location'].dropna():
            for city in loc.split(','):
                cities.add(city.strip())
        return sorted(list(cities))
    return []

df = load_data()

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


# =========================================== Page d'accueil ===============================================
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

# ====================================== Page de création de profil =====================================================

elif st.session_state.page == "profile":
    render_navbar()
    
    st.markdown("## 👤 Créez votre profil professionnel")
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Votre nom", value=st.session_state.get("name", ""))
            email = st.text_input("Votre email", value=st.session_state.get("email", ""))
        
        with col2:
            if df is not None:
                unique_cities = get_unique_cities(df)

            location_index = 0
            if st.session_state.get("selected_location") and st.session_state.get("selected_location") in unique_cities:
                location_index = unique_cities.index(st.session_state.get("selected_location"))
  
            location = st.selectbox("Votre ville", unique_cities, index=location_index)

            unique_sectors = get_unique_sectors(df)
            sector_index = 0
            if st.session_state.get("selected_sector") and st.session_state.get("selected_sector") in unique_sectors:
                sector_index = unique_sectors.index(st.session_state["selected_sector"])

            selected_sector = st.selectbox("Secteur d'activité", unique_sectors, index=sector_index)
        
        st.markdown("### Vos compétences")
        if df is not None:
            skills = get_unique_skills(df)
        else:
            skills = []

        # Sécurité : n'afficher que les compétences valides comme valeurs par défaut
        default_skills = [s for s in st.session_state.get("selected_skills", []) if s in skills]

        selected_skills = st.multiselect(
            "Sélectionnez vos compétences", 
            skills
            )

        
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
                st.session_state.selected_sector = selected_sector
                st.session_state.profile_complete = True
                
                try:
                    with engine.begin() as conn:
                    # Vérifier si l'utilisateur existe déjà
                        result = conn.execute(
                            text("SELECT id FROM users WHERE username = :username"),
                            {"username": name}
                        ).fetchone()
        
                        if result:
                            user_id = result[0]
                            # Mise à jour du profil existant
                            conn.execute(text("""
                                UPDATE users
                                SET email = :email,
                                    location = :location,
                                    skills = :skills
                                WHERE id = :id
                            """), {
                                "email": email,
                                "location": location,
                                "skills": json.dumps(selected_skills),
                                "id": user_id
                            })
                        else:
                            # Insertion d'un nouvel utilisateur
                            conn.execute(text("""
                                INSERT INTO users (username, email, location, skills)
                                VALUES (:username, :email, :location, :skills)
                            """), {
                                "username": name,
                                "email": email,
                                "location": location,
                                "skills": json.dumps(selected_skills)
                            })
                            st.success(f"✅ INSERT exécuté avec : {name}, {email}, {location}")

                            user_id = conn.execute(
                                text("SELECT id FROM users WHERE username = :username"),
                                {"username": name}
                            ).fetchone()[0]

                            st.markdown(f"""
                                ### 👤 Profil enregistré :
                                - **Nom :** {name}
                                - **Email :** {email}
                                - **Ville :** {location}
                                - **Compétences :** {', '.join(selected_skills)}
                            """)
                        st.session_state.user_id = user_id
                        st.session_state.name = name
                        st.session_state.index = 0
                
                        st.success("✅ Profil enregistré avec succès !")
                        st.balloons()
                
                except Exception as e:
                    st.error(f"Erreur SQL : {e}")
                    print("Erreur SQL : ", e)
                    st.error(f"❌ Erreur lors de l'enregistrement dans Supabase : {e}")

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
        job = st.session_state.current_job
        jobs = [job]

        
        if not job:
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
            
                
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💖 Voir mes matchs", type="primary", use_container_width=True):
                    st.session_state.page = "matches"
                    st.rerun()
            with col2:
                if st.button("⚙️ Modifier mon profil", use_container_width=True):
                    st.session_state.page = "profile"
                    st.rerun()

            # Affichage de l'offre actuelle
            job = job  # déjà un dictionnaire retourné par get_random_offer()

            # Carte de l'offre
            location = job.get("location", "Non précisée")
            category = job.get("job_category", "Non Précisé")
            company = job.get("company", "Non précisé")
            title = job.get("cleaned_title", "Titre non précisé")
            sector = job.get("secteur_activite", "Non précisé")
            description = job.get("description", "Aucune description disponible.")


            st.markdown("""
                    <div class="job-card">
                        <div class="match-badge">✨ Votre Meilleur Match !</div>
                        <h2 style="margin-bottom: 0.5rem; color: #1F2937;">{title}</h2>
                        <p class="company-name" style="margin-bottom: 1.5rem;">{company}</p>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                            <div>
                                <p style="color: #6B7280; margin-bottom: 0.2rem; font-weight: 500;">📍 Localisation</p>
                                <p style="color: #374151; font-weight: 600;">{location}</p>
                            </div>
                            <div>
                                <p style="color: #6B7280; margin-bottom: 0.2rem; font-weight: 500;">📅 Type de Job</p>
                                <p style="color: #374151; font-weight: 600;">{category}</p>
                            </div>
                            <div>
                                <p style="color: #6B7280; margin-bottom: 0.2rem; font-weight: 500;">🏢 Secteur d'activité</p>
                                <p style="color: #374151; font-weight: 600;">{sector}</p>
                            </div>
                        </div>
                        <div style="margin-top: 1.5rem;">
                            <h4 style="color: #1F2937;">📝 Description de l'offre</h4>
                            <p style="color: #374151; line-height: 1.6;">{description}</p>
                        </div>
                    </div>  
            """.format(title=title, company=company, location=location, category=category, description=description, sector=sector), unsafe_allow_html=True)
        
            # Extraction et affichage des compétences issues de 'final_skills'
            final_skills = job.get("final_skills", "")
            skills_list = [s.strip() for s in final_skills.split(",") if s.strip()]

            if skills_list:
                st.markdown("#### 🧠 Compétences requises")
                html_skills = "<div style='margin-top: 0.5rem;'>"

                for skill in skills_list[:10]:
                    html_skills += "<span style='display: inline-block; background-color: #EFF6FF; color: #1D4ED8; border-radius: 12px; padding: 6px 12px; margin: 4px 6px 4px 0; font-size: 0.85rem; font-weight: 500;'>{}</span>".format(skill)

                html_skills += "</div>"
                st.markdown(html_skills, unsafe_allow_html=True)
            # Boutons de swipe
            col1, col2, col3 = st.columns([1, 1, 1])
                
            # ===================== Bloc de feedback intermédiaire =============================
            with col1:
                if st.button("👎 Passer", use_container_width=True):
                    st.session_state.current_job = get_random_offer(engine)
                    st.rerun()

            # Bouton J'aime qui active le feedback
            with col3:
                if st.button("👍 J'aime", use_container_width=True, type="primary"):
                    st.session_state.pending_like_job = job
                    st.session_state.feedback_step = True

            if st.session_state.get("feedback_step", False):
                job_dict = st.session_state.get("pending_like_job")
                st.success("💖 Vous avez aimé cette offre. Dites-nous pourquoi :")

                choix = st.multiselect("Qu'avez-vous apprécié dans cette offre ?", [
                    "Les compétences requises",
                    "La localisation",
                    "Le secteur d'activité"
                ])

                if st.button("✅ Valider mes préférences"):
                    feedback = {}
                    key_map = {
                        "Les compétences requises": "final_skills",
                        "La localisation": "location",
                        "Le secteur d'activité": "secteur_activite"
                    }

                    for r in choix:
                        if r in key_map:
                            feedback[key_map[r]] = "👍 J'aime"

                    job_dict = st.session_state.get("pending_like_job")
                    user_id = st.session_state.get("user_id")
                    offer_id = job_dict.get("job_id") or job_dict.get("id")

                    # ✅ Insérer dans Supabase
                    with engine.begin() as conn:
                        conn.execute(text("""
                                INSERT INTO interactions (
                                    user_id, offer_id,
                                    feedback_competence,
                                    feedback_ville, feedback_secteur
                                ) VALUES (
                                    :user_id, :offer_id,
                                    :fb_skills, :fb_location, :fb_secteur
                                )
                        """), {
                                "user_id": user_id,
                                "offer_id": offer_id,
                                "fb_skills": feedback.get("final_skills") is not None,
                                "fb_location": feedback.get("location") is not None,
                                "fb_secteur": feedback.get("secteur_activite") is not None
                        })

                        # Ajouter à la liste des matchs (local)
                        if "matches" not in st.session_state:
                            st.session_state.matches = []

                        st.session_state.matches.append(job_dict)
                        st.success("✅ Vos préférences ont été enregistrées dans Supabase !")
                        st.session_state.feedback_step = False
                        st.session_state.pending_like_job = None
                        st.session_state.current_job = get_random_offer(engine)
                        st.rerun()

                st.stop()

                

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
                col1, col2 = st.columns([3, 1])
                with col1:
                    title = match.get("title", "Titre non précisé")
                    company = match.get("company", "Entreprise non précisée")
                    location = match.get("location", "Non précisée")
                    category = match.get("job_category", "Non précisé")
                    description = match.get("description", "Aucune description disponible.")
                    sector = match.get("secteur_activite", "Non précisé")

                    st.markdown("""
                        <div class="job-card">
                            <div class="match-badge">✨ Votre Meilleur Match !</div>
                            <h2 style="margin-bottom: 0.5rem; color: #1F2937;">{title}</h2>
                            <p class="company-name" style="margin-bottom: 1.5rem;">{company}</p>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                                <div>
                                    <p style="color: #6B7280; margin-bottom: 0.2rem; font-weight: 500;">📍 Localisation</p>
                                    <p style="color: #374151; font-weight: 600;">{location}</p>
                                </div>
                                <div>
                                    <p style="color: #6B7280; margin-bottom: 0.2rem; font-weight: 500;">📅 Type de Job</p>
                                    <p style="color: #374151; font-weight: 600;">{category}</p>
                                </div>
                                <div>
                                    <p style="color: #6B7280; margin-bottom: 0.2rem; font-weight: 500;">🏢 Secteur d'activité</p>
                                    <p style="color: #374151; font-weight: 600;">{sector}</p>
                                </div>
                            </div>
                        </div>  
                """.format(title=title, company=company, location=location, category=category, description=description, sector=sector), unsafe_allow_html=True)


                    # Extraire et afficher les compétences
                    final_skills = match.get("final_skills", "")
                    skills_list = [s.strip() for s in final_skills.split(",") if s.strip()]

                    if skills_list:
                        st.markdown("#### 🧠 Compétences requises")
                        html_skills = '<div style="margin-top: 0.5rem;">'
    
                        for skill in skills_list[:10]:  # Limite à 10 compétences affichées
                            html_skills += f"<span style='display: inline-block; background-color: #EFF6FF; color: #1D4ED8; border-radius: 12px; padding: 6px 12px; margin: 4px 6px 4px 0; font-size: 0.85rem;'>{skill}</span>"
    
                        html_skills += "</div>"
                        st.markdown(html_skills, unsafe_allow_html=True)
                        
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
