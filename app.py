import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import tempfile

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Conseiller Métiers du Numérique",
    page_icon="💻",
    layout="centered"
)

# ── CSS personnalisé ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .profile-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .badge-scolaire { background-color: #d0f0c0; color: #2d6a2d; }
    .badge-emploi { background-color: #cce5ff; color: #004085; }
    .badge-reconversion { background-color: #fff3cd; color: #856404; }
    .badge-indefini { background-color: #e2e3e5; color: #383d41; }
    h1 { color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ── Profils disponibles ────────────────────────────────────────────────────────
PROFILS = {
    "scolaire": {
        "label": "🎓 Élève / Étudiant",
        "badge_class": "badge-scolaire",
        "system": """Tu es un conseiller expert en orientation scolaire vers les métiers du numérique.
Tu t'adresses à des élèves et étudiants. Ton ton est encourageant, accessible et motivant.
Tu proposes des pistes de formations, de diplômes, et d'expériences pratiques.
Réponds toujours en français."""
    },
    "emploi": {
        "label": "🔍 Demandeur d'emploi",
        "badge_class": "badge-emploi",
        "system": """Tu es un conseiller emploi spécialisé dans les métiers du numérique.
Tu aides les personnes en recherche d'emploi à identifier les métiers porteurs, les compétences recherchées et les formations rapides.
Ton ton est professionnel, pragmatique et bienveillant.
Réponds toujours en français."""
    },
    "reconversion": {
        "label": "🔄 Cadre en reconversion",
        "badge_class": "badge-reconversion",
        "system": """Tu es un coach expert en reconversion professionnelle vers le numérique.
Tu t'adresses à des cadres et professionnels expérimentés. Tu valorises leurs compétences transverses.
Tu proposes des passerelles métier réalistes et des plans de transition concrets.
Réponds toujours en français."""
    },
    "indefini": {
        "label": "❓ Profil à définir",
        "badge_class": "badge-indefini",
        "system": """Tu es un conseiller en orientation générale sur les métiers du numérique.
Tu poses des questions pour mieux comprendre la situation de l'utilisateur avant de l'orienter.
Sois curieux, bienveillant et progressif dans tes questions.
Réponds toujours en français."""
    }
}

PROMPT_DETECTION_PROFIL = """
Analyse ce message et détermine le profil de l'utilisateur parmi : scolaire, emploi, reconversion, indefini.
- scolaire : élève, étudiant, lycéen, en études, orientation scolaire
- emploi : demandeur d'emploi, cherche du travail, chômage, reconversion rapide
- reconversion : cadre, manager, professionnel expérimenté souhaitant changer de métier
- indefini : pas assez d'éléments

Message : "{message}"

Réponds uniquement avec un mot parmi : scolaire, emploi, reconversion, indefini
"""

# ── Fonctions utilitaires ──────────────────────────────────────────────────────

def detect_profil(message: str, llm) -> str:
    """Détecte le profil utilisateur via le LLM."""
    try:
        prompt = PROMPT_DETECTION_PROFIL.format(message=message)
        response = llm.invoke(prompt)
        profil = response.content.strip().lower()
        if profil in PROFILS:
            return profil
        # Détection par mots-clés de secours
        msg = message.lower()
        if any(k in msg for k in ["étudiant", "lycée", "école", "université", "bac", "études"]):
            return "scolaire"
        if any(k in msg for k in ["emploi", "travail", "chômage", "recrutement", "cv"]):
            return "emploi"
        if any(k in msg for k in ["reconversion", "cadre", "manager", "expérience", "changer"]):
            return "reconversion"
    except:
        pass
    return "indefini"


def load_documents(uploaded_files):
    """Charge et découpe les documents PDF/Word uploadés."""
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for uploaded_file in uploaded_files:
        suffix = "." + uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            if suffix == ".pdf":
                loader = PyMuPDFLoader(tmp_path)
            elif suffix in [".docx", ".doc"]:
                loader = Docx2txtLoader(tmp_path)
            else:
                continue
            raw_docs = loader.load()
            chunks = splitter.split_documents(raw_docs)
            docs.extend(chunks)
        except Exception as e:
            st.warning(f"Erreur lors du chargement de {uploaded_file.name} : {e}")
        finally:
            os.unlink(tmp_path)

    return docs


def build_vectorstore(docs, embeddings):
    """Construit la base vectorielle ChromaDB."""
    return Chroma.from_documents(docs, embeddings)


def build_chain(vectorstore, llm, profil_key: str):
    """Construit la chaîne RAG conversationnelle."""
    profil = PROFILS[profil_key]

    prompt_template = profil["system"] + """

Utilise les extraits de documents suivants pour répondre à la question.
Si les documents ne contiennent pas la réponse, utilise tes connaissances générales sur les métiers du numérique.

Contexte documentaire :
{context}

Historique de la conversation :
{chat_history}

Question : {question}

Réponse :"""

    qa_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=prompt_template
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=False,
        verbose=False
    )
    return chain


# ── Initialisation session state ───────────────────────────────────────────────
def init_session():
    defaults = {
        "messages": [],
        "profil": None,
        "chain": None,
        "vectorstore": None,
        "docs_loaded": False,
        "api_key_ok": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Interface principale ───────────────────────────────────────────────────────
def main():
    init_session()

    st.title("💻 Conseiller Métiers du Numérique")
    st.caption("Un assistant IA pour explorer et choisir votre voie dans le numérique")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Clé API Gemini
        api_key = st.text_input(
            "🔑 Clé API Google Gemini",
            type="password",
            placeholder="AIza...",
            help="Obtenez votre clé gratuite sur https://aistudio.google.com"
        )

        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.session_state.api_key_ok = True
            st.success("✅ Clé API configurée")

        st.divider()

        # Upload documents
        st.subheader("📄 Vos documents")
        uploaded_files = st.file_uploader(
            "Ajoutez vos PDF / Word",
            type=["pdf", "docx", "doc"],
            accept_multiple_files=True,
            help="Ces documents enrichiront les réponses du conseiller"
        )

        if uploaded_files and st.session_state.api_key_ok:
            if st.button("📥 Charger les documents", use_container_width=True):
                with st.spinner("Indexation en cours..."):
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-2.0-flash",
                            temperature=0.3,
                            google_api_key=api_key
                        )
                        embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/embedding-001",
                            google_api_key=api_key
                        )
                        docs = load_documents(uploaded_files)
                        if docs:
                            st.session_state.vectorstore = build_vectorstore(docs, embeddings)
                            st.session_state.docs_loaded = True
                            st.session_state.llm = llm
                            st.success(f"✅ {len(docs)} extraits indexés !")
                        else:
                            st.error("Aucun document valide trouvé.")
                    except Exception as e:
                        st.error(f"Erreur : {e}")

        st.divider()

        # Profil manuel
        st.subheader("👤 Profil détecté")
        if st.session_state.profil:
            p = PROFILS[st.session_state.profil]
            st.markdown(
                f'<span class="profile-badge {p["badge_class"]}">{p["label"]}</span>',
                unsafe_allow_html=True
            )
        else:
            st.info("Profil auto-détecté à la première question")

        if st.button("🔄 Réinitialiser la conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.profil = None
            st.session_state.chain = None
            st.rerun()

        st.divider()
        st.caption("POC — Chatbot Métiers du Numérique\nPowered by Gemini Flash 2.0 + RAG")

    # ── Zone principale ────────────────────────────────────────────────────────
    if not st.session_state.api_key_ok:
        st.info("👈 Commencez par entrer votre clé API Gemini dans le panneau de gauche.")
        st.markdown("""
        **Comment obtenir une clé gratuite ?**
        1. Rendez-vous sur [Google AI Studio](https://aistudio.google.com)
        2. Connectez-vous avec votre compte Google
        3. Cliquez sur **"Get API Key"**
        4. Copiez-collez la clé ici
        """)
        return

    # Message d'accueil
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("""
Bonjour ! 👋 Je suis votre **conseiller en métiers du numérique**.

Je suis là pour vous aider à :
- 🎓 Découvrir les métiers du numérique selon votre profil
- 💡 Obtenir des conseils personnalisés d'orientation
- 🚀 Construire votre projet professionnel dans le secteur tech

**Dites-moi qui vous êtes et ce que vous recherchez !**
*(exemple : "Je suis étudiant en terminale et je cherche ma voie dans l'informatique")*
            """)

    # Affichage historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input utilisateur
    if user_input := st.chat_input("Posez votre question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Réflexion en cours..."):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash",
                        temperature=0.3,
                        google_api_key=api_key
                    )

                    # Détection du profil au premier message
                    if not st.session_state.profil:
                        profil = detect_profil(user_input, llm)
                        st.session_state.profil = profil
                        p = PROFILS[profil]
                        st.markdown(
                            f'<span class="profile-badge {p["badge_class"]}">Profil détecté : {p["label"]}</span>',
                            unsafe_allow_html=True
                        )

                    # Construction/récupération de la chaîne RAG
                    if st.session_state.chain is None:
                        if st.session_state.vectorstore:
                            st.session_state.chain = build_chain(
                                st.session_state.vectorstore,
                                llm,
                                st.session_state.profil
                            )
                        else:
                            # Sans documents : LLM seul avec prompt profil
                            profil_info = PROFILS[st.session_state.profil]
                            system = profil_info["system"]
                            history = "\n".join([
                                f"{m['role'].capitalize()} : {m['content']}"
                                for m in st.session_state.messages[:-1]
                            ])
                            full_prompt = f"{system}\n\nHistorique :\n{history}\n\nQuestion : {user_input}\n\nRéponse :"
                            response = llm.invoke(full_prompt)
                            answer = response.content
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            return

                    # Appel RAG
                    result = st.session_state.chain({"question": user_input})
                    answer = result["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    err_msg = f"❌ Erreur : {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})


if __name__ == "__main__":
    main()
