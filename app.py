import streamlit as st
import os
import tempfile

import google.generativeai as genai
from chromadb import Client
from chromadb.config import Settings
import chromadb

# ── Loaders ────────────────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx2txt
except ImportError:
    docx2txt = None

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Conseiller Métiers du Numérique",
    page_icon="💻",
    layout="centered"
)

st.markdown("""
<style>
    .profile-badge { display:inline-block; padding:4px 14px; border-radius:20px; font-size:13px; font-weight:600; margin-bottom:12px; }
    .badge-scolaire { background-color:#d0f0c0; color:#2d6a2d; }
    .badge-emploi { background-color:#cce5ff; color:#004085; }
    .badge-reconversion { background-color:#fff3cd; color:#856404; }
    .badge-indefini { background-color:#e2e3e5; color:#383d41; }
</style>
""", unsafe_allow_html=True)

# ── Profils ────────────────────────────────────────────────────────────────────
PROFILS = {
    "scolaire": {
        "label": "🎓 Élève / Étudiant",
        "badge_class": "badge-scolaire",
        "system": "Tu es un conseiller expert en orientation scolaire vers les métiers du numérique. Tu t'adresses à des élèves et étudiants. Ton ton est encourageant, accessible et motivant. Tu proposes des pistes de formations, diplômes et expériences pratiques. Réponds toujours en français."
    },
    "emploi": {
        "label": "🔍 Demandeur d'emploi",
        "badge_class": "badge-emploi",
        "system": "Tu es un conseiller emploi spécialisé dans les métiers du numérique. Tu aides les personnes en recherche d'emploi à identifier les métiers porteurs, compétences recherchées et formations rapides. Ton ton est professionnel et bienveillant. Réponds toujours en français."
    },
    "reconversion": {
        "label": "🔄 Cadre en reconversion",
        "badge_class": "badge-reconversion",
        "system": "Tu es un coach expert en reconversion professionnelle vers le numérique. Tu valorises les compétences transverses des cadres. Tu proposes des passerelles métier réalistes et des plans de transition concrets. Réponds toujours en français."
    },
    "indefini": {
        "label": "❓ Profil à définir",
        "badge_class": "badge-indefini",
        "system": "Tu es un conseiller en orientation générale sur les métiers du numérique. Pose des questions pour mieux comprendre la situation de l'utilisateur avant de l'orienter. Sois curieux et bienveillant. Réponds toujours en français."
    }
}

# ── Fonctions utilitaires ──────────────────────────────────────────────────────

def extract_text_from_pdf(filepath):
    if fitz is None:
        return ""
    text = ""
    doc = fitz.open(filepath)
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(filepath):
    if docx2txt is None:
        return ""
    return docx2txt.process(filepath)

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]

def get_embedding(text, api_key):
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return result["embedding"]

def get_query_embedding(text, api_key):
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query"
    )
    return result["embedding"]

def build_vectorstore(texts, api_key):
    client = chromadb.Client()
    collection = client.create_collection("docs")
    for i, text in enumerate(texts):
        emb = get_embedding(text, api_key)
        collection.add(embeddings=[emb], documents=[text], ids=[f"chunk_{i}"])
    return collection

def search_docs(collection, query, api_key, n=4):
    emb = get_query_embedding(query, api_key)
    results = collection.query(query_embeddings=[emb], n_results=min(n, collection.count()))
    return results["documents"][0] if results["documents"] else []

def detect_profil(message, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Analyse ce message et détermine le profil parmi : scolaire, emploi, reconversion, indefini.
- scolaire : élève, étudiant, lycéen, en études, orientation
- emploi : demandeur d'emploi, cherche du travail, chômage
- reconversion : cadre, manager, professionnel expérimenté, changer de métier
- indefini : pas assez d'éléments
Message : "{message}"
Réponds uniquement avec un mot parmi : scolaire, emploi, reconversion, indefini"""
        response = model.generate_content(prompt)
        profil = response.text.strip().lower()
        if profil in PROFILS:
            return profil
    except:
        pass
    msg = message.lower()
    if any(k in msg for k in ["étudiant", "lycée", "école", "université", "bac", "études"]):
        return "scolaire"
    if any(k in msg for k in ["emploi", "travail", "chômage", "cv", "recrutement"]):
        return "emploi"
    if any(k in msg for k in ["reconversion", "cadre", "manager", "expérience", "changer"]):
        return "reconversion"
    return "indefini"

def generate_response(user_input, profil_key, history, context_docs, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    profil = PROFILS[profil_key]

    context = "\n\n".join(context_docs) if context_docs else "Aucun document disponible."
    hist_text = "\n".join([f"{m['role'].capitalize()} : {m['content']}" for m in history[-6:]])

    prompt = f"""{profil['system']}

Extraits de documents pertinents :
{context}

Historique de la conversation :
{hist_text}

Question de l'utilisateur : {user_input}

Réponse (en français) :"""

    response = model.generate_content(prompt)
    return response.text

def load_folder_docs(api_key):
    folder = "docs"
    all_chunks = []
    filenames = []
    if not os.path.exists(folder):
        return [], []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        ext = filename.lower().split(".")[-1]
        text = ""
        if ext == "pdf":
            text = extract_text_from_pdf(filepath)
        elif ext in ["docx", "doc"]:
            text = extract_text_from_docx(filepath)
        if text.strip():
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            filenames.append(filename)
    return all_chunks, filenames

def load_uploaded_docs(uploaded_files):
    all_chunks = []
    for uf in uploaded_files:
        suffix = "." + uf.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        text = ""
        if suffix == ".pdf":
            text = extract_text_from_pdf(tmp_path)
        elif suffix in [".docx", ".doc"]:
            text = extract_text_from_docx(tmp_path)
        os.unlink(tmp_path)
        if text.strip():
            all_chunks.extend(chunk_text(text))
    return all_chunks

# ── Init session ───────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [], "profil": None, "collection": None,
        "docs_loaded": False, "api_key_ok": False,
        "preloaded_files": [], "preload_done": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    init_session()
    st.title("💻 Conseiller Métiers du Numérique")
    st.caption("Assistant IA de conseil et coaching sur les métiers du numérique")

    with st.sidebar:
        st.header("⚙️ Configuration")

        # Clé API
        api_key = None
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            st.session_state.api_key_ok = True
            st.success("✅ Clé API Gemini active")
        except:
            api_key = st.text_input("🔑 Clé API Google Gemini", type="password", placeholder="AIza...")
            if api_key:
                st.session_state.api_key_ok = True
                st.success("✅ Clé API configurée")

        # Pré-chargement docs/
        if api_key and not st.session_state.preload_done:
            with st.spinner("📚 Chargement des documents..."):
                try:
                    chunks, files = load_folder_docs(api_key)
                    if chunks:
                        st.session_state.collection = build_vectorstore(chunks, api_key)
                        st.session_state.docs_loaded = True
                        st.session_state.preloaded_files = files
                except Exception as e:
                    st.warning(f"Erreur docs : {e}")
                finally:
                    st.session_state.preload_done = True

        st.divider()
        st.subheader("📄 Base de connaissance")
        if st.session_state.preloaded_files:
            st.success(f"✅ {len(st.session_state.preloaded_files)} document(s) chargé(s)")
            for f in st.session_state.preloaded_files:
                st.caption(f"📎 {f}")
        else:
            st.info("Dossier `docs/` vide — mode LLM seul")

        with st.expander("➕ Ajouter des documents"):
            uploaded_files = st.file_uploader("PDF / Word", type=["pdf", "docx", "doc"], accept_multiple_files=True)
            if uploaded_files and api_key:
                if st.button("📥 Indexer", use_container_width=True):
                    with st.spinner("Indexation..."):
                        try:
                            chunks = load_uploaded_docs(uploaded_files)
                            if chunks:
                                st.session_state.collection = build_vectorstore(chunks, api_key)
                                st.session_state.docs_loaded = True
                                st.success(f"✅ {len(chunks)} extraits indexés !")
                        except Exception as e:
                            st.error(f"Erreur : {e}")

        st.divider()
        st.subheader("👤 Profil détecté")
        if st.session_state.profil:
            p = PROFILS[st.session_state.profil]
            st.markdown(f'<span class="profile-badge {p["badge_class"]}">{p["label"]}</span>', unsafe_allow_html=True)
        else:
            st.info("Auto-détecté à la 1ère question")

        if st.button("🔄 Réinitialiser la conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.profil = None
            st.rerun()

        st.divider()
        st.caption("POC — Chatbot Métiers du Numérique\n🤖 Gemini Flash 2.0 + RAG")

    # Zone principale
    if not st.session_state.api_key_ok:
        st.info("👈 Entrez votre clé API Gemini dans le panneau de gauche.")
        st.markdown("Clé gratuite sur [aistudio.google.com](https://aistudio.google.com)")
        return

    # Message d'accueil
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("""Bonjour ! 👋 Je suis votre **conseiller en métiers du numérique**.

Je suis là pour vous aider à :
- 🎓 Découvrir les métiers du numérique selon votre profil
- 💡 Obtenir des conseils personnalisés d'orientation
- 🚀 Construire votre projet professionnel dans le secteur tech

**Dites-moi qui vous êtes et ce que vous recherchez !**
*(ex : "Je suis étudiant en terminale et je cherche ma voie dans l'informatique")*""")

    # Historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if user_input := st.chat_input("Posez votre question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Réflexion en cours..."):
                try:
                    # Détection profil
                    if not st.session_state.profil:
                        profil = detect_profil(user_input, api_key)
                        st.session_state.profil = profil
                        p = PROFILS[profil]
                        st.markdown(f'<span class="profile-badge {p["badge_class"]}">Profil détecté : {p["label"]}</span>', unsafe_allow_html=True)

                    # Recherche RAG
                    context_docs = []
                    if st.session_state.collection:
                        context_docs = search_docs(st.session_state.collection, user_input, api_key)

                    # Génération réponse
                    answer = generate_response(
                        user_input,
                        st.session_state.profil,
                        st.session_state.messages[:-1],
                        context_docs,
                        api_key
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    err_msg = f"❌ Erreur : {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

if __name__ == "__main__":
    main()
