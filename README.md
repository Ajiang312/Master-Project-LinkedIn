# 💼 Master Project – LinkedIn Job Matching & Recommendation

Ce projet vise à développer une solution intelligente de recommandation d’offres d’emploi pour les profils techniques (ex. ingénieurs), à partir des compétences et de la localisation extraites de données LinkedIn.

---

## 🗂️ Structure du projet

| Dossier                     | Description |
|----------------------------|-------------|
| `1 - Collecte Donnees`     | Scripts de scraping des offres LinkedIn |
| `2 - Nettoyage`            | Nettoyage des descriptions, suppression d'emojis, stopwords |
| `3 - Traitement`           | Normalisation des compétences, transformation des données |
| `4 - ML`                   | Modèles de recommandation, vectorisation, matching |
| `5 - PipeLine Final`       | Assemblage final (interface, prédiction, feedback utilisateur) |
| `.gitignore`               | Fichiers ignorés par Git |

---

## 👥 Répartition du travail

| Membre   | Responsabilités principales |
|----------|-----------------------------|
| **Antoine** | - Scraping des offres d’emploi (Playwright)<br>- Nettoyage et traitement des données (1, 2, 3) |
| **Roman**   | - Développement des modèles ML (recommandation)<br>- Import des données dans Supabase (PostgreSQL) |
| **Tess**    | - Création de l’interface utilisateur (Streamlit / HTML-CSS)<br>- Affichage dynamique, gestion des interactions utilisateur |

---

## 🎯 Objectifs du projet

- Collecter des données d’offres d’emploi techniques depuis LinkedIn.
- Extraire les compétences et la localisation à partir des descriptions.
- Créer un moteur de recommandation basé sur les compétences et la ville.
- Concevoir une interface inspirée de Tinder (swipe gauche/droite).
- Intégrer un système de feedback utilisateur pour ajuster les recommandations.

---

## ⚙️ Technologies utilisées

| Technologie | Usage |
|-------------|-------|
| **Python** | Traitement de données, Machine Learning |
| **Playwright** | Scraping LinkedIn |
| **Scikit-learn** | Modélisation et vectorisation |
| **PostgreSQL** + **Supabase** | Stockage des données |
| **Streamlit** / **HTML-CSS** | Interface utilisateur |
| **Git & GitHub** | Suivi de version et collaboration |

---

## 🚀 Perspectives d'amélioration

- Apprentissage des préférences utilisateurs via feedback
- Ajout de filtres avancés (salaire, niveau d’expérience, secteur)
- Déploiement Dockerisé sur Google Cloud Run
- Affinage du matching (fuzzy matching, pondération des skills)

---

> 🎓 Projet réalisé dans le cadre de la Majeure Big Data & Marketing Digital – ESME Sudria (2025).
