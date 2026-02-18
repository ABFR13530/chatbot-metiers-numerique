# 💻 POC Chatbot — Conseiller Métiers du Numérique

Chatbot IA de conseil et coaching sur les métiers du numérique, avec détection automatique de profil et RAG sur vos documents.

---

## 🚀 Déploiement sur Streamlit Cloud (lien public gratuit)

### Étape 1 — Préparer GitHub
1. Créez un compte sur [github.com](https://github.com) si vous n'en avez pas
2. Créez un **nouveau repository public** (ex: `chatbot-metiers-numerique`)
3. Uploadez ces 3 fichiers :
   - `app.py`
   - `requirements.txt`
   - `README.md`

### Étape 2 — Déployer sur Streamlit Cloud
1. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io)
2. Connectez-vous avec GitHub
3. Cliquez **"New app"**
4. Sélectionnez votre repo → branche `main` → fichier `app.py`
5. Cliquez **"Deploy"** → votre lien public est généré en ~2 minutes !

### Étape 3 — Obtenir votre clé API Gemini (gratuite)
1. Allez sur [aistudio.google.com](https://aistudio.google.com)
2. Connectez-vous avec Google
3. Cliquez **"Get API Key"** → **"Create API Key"**
4. Copiez la clé → collez-la dans l'interface du chatbot

---

## 🧠 Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| 🎯 Détection de profil | Auto-détecte scolaire / demandeur d'emploi / cadre en reconversion |
| 📄 RAG sur documents | Ingère vos PDF et Word pour des réponses contextualisées |
| 💬 Mémoire conversationnelle | Se souvient du contexte tout au long de la session |
| 🇫🇷 100% Français | Interface et réponses en français |
| ☁️ 100% gratuit | Gemini Flash 2.0 + Streamlit Cloud = 0€ |

---

## 👤 Profils gérés

- 🎓 **Scolaire** — Élèves, étudiants, orientation post-bac
- 🔍 **Demandeur d'emploi** — Métiers porteurs, formations rapides
- 🔄 **Cadre en reconversion** — Passerelles métier, valorisation de l'expérience
- ❓ **Indéfini** — Questions de clarification avant orientation

---

## 🏗️ Architecture

```
Utilisateur
    ↓
Streamlit (interface)
    ↓
Détection profil (Gemini Flash 2.0)
    ↓
RAG : ChromaDB + vos documents
    ↓
Gemini Flash 2.0 (réponse personnalisée)
    ↓
Réponse affichée
```

---

## 📁 Structure des fichiers

```
├── app.py              # Application principale
├── requirements.txt    # Dépendances Python
└── README.md           # Ce fichier
```

---

## ⚡ Usage sans documents

Le chatbot fonctionne aussi **sans documents** : il utilise alors les connaissances générales de Gemini sur les métiers du numérique. L'ajout de documents enrichit et personnalise les réponses.

---

*POC réalisé avec Streamlit + LangChain + Gemini Flash 2.0 + ChromaDB*
