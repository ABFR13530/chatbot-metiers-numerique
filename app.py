import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import tempfile

st.set_page_config(page_title="Conseiller Métiers du Numérique", page_icon="💻", layout="centered")

st.markdown("""
<style>
    .profile-badge { display:inline-block; padding:4px 12px; border-radius:20px; font-size:13px; font-weight:600; margin-bottom:12px; }
    .badge-scolaire { background-color:#d0f0c0; color:#2d6a2d; }
    .badge-emploi { background-color:#cce5ff; color:#004085; }
    .badge-reconversion { background-color:#fff3cd; color:#856404; }
    .badge-indefini { background-color:#e2e3e5; color:#383d41; }
</style>
""", unsafe_allow_html=True)

PROFILS = {
    "scolaire": {
        "label": "🎓 Élève / Étudiant", "badge_class": "badge-scolaire",
        "system": "Tu es un conseiller expert en orientation scolaire vers les métiers du numérique. Tu t'adresses à des élèves et étudiants. Ton ton est encourageant, accessible et motivant. Tu proposes des pistes de formations, de diplômes, et d'expériences pratiques. Réponds toujours en français."
    },
    "emploi": {
        "label": "🔍 Demandeur d'emploi", "badge_class": "badge-emploi",
        "system": "Tu es un conseiller emploi spécialisé dans les métiers du numérique. Tu aides les personnes en recherche d'emploi à identifier les métiers porteurs, les compétences recherchées et les formations rapides. Ton ton est professionnel, pragmatique et bienveillant. Réponds toujours en français."
    },
    "reconversion": {
        "label": "🔄 Cadre en reconversion", "badge_class": "badge-reconversion",
        "system": "Tu es un coach expert en reconversion professionnelle vers le numérique. Tu t'adresses à des cadres et professionnels expérimentés. Tu valorises leurs compétences transverses. Tu proposes des passerelles métier réalistes et des plans de transition concrets. Réponds toujours en français."
    },
    "indefini": {
        "label": "❓ Profil à définir", "badge_class": "badge-indefini",
        "system": "Tu es un conseiller en orientation générale sur les métiers du numérique. Tu poses des questions pour mieux comprendre la situation de l'utilisateur avant de l'orienter. Sois curieux, bienveillant et progressif dans tes questions. Réponds toujours en français."
    }
}

def detect_profil(message, llm):
    try:
        prompt = f"""Analyse ce message et détermine le profil parmi : scolaire, emploi, reconversion, indefini.
- scolaire : élève, étudiant, lycéen, en études
- emploi : demandeur d'emploi, cherche du travail, chômage
- reconversion : cadre, manager, professionnel expérimenté souhaitant changer
- indefini : pas assez d'éléments
Message : "{message}"
Réponds uniquement avec un mot parmi : scolaire, emploi, reconversion, indefini"""
        response = llm.invoke(prompt)
        profil = response.content.strip().lower()
        if profil in PROFILS:
            return profil
    except:
        pass
    msg = message.lower()
    if any(k in msg for k in ["étudiant", "lycée", "école", "université", "bac"]):
        return "scolaire"
    if any(k in msg for k in ["emploi", "travail", "chômage", "cv"]):
        return "emploi"
    if any(k in msg for k in ["reconversion", "cadre", "manager", "expérience"]):
        return "reconversion"
    return "indefini"

def load_docs_from_folder(embeddings):
    folder_path = "docs"
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    if not os.path.exists(folder_path):
        return None, 0, []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".pdf", ".docx", ".doc"))]
    if not files:
        return None, 0, []
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            loader = PyMuPDFLoader(filepath) if filename.lower().endswith(".pdf") else Docx2txtLoader(filepath)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(loader.load())
            docs.extend(chunks)
        except Exception as e:
            st.warning(f"Erreur {filename} : {e}")
    if docs:
        return Chroma.from_documents(docs, embeddings), len(docs), files
    return None, 0, []

def load_documents_from_upload(uploaded_files):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for uf in uploaded_files:
        suffix = "." + uf.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.read())
            tmp_path = tmp.name
        try:
            loader = PyMuPDFLoader(tmp_path) if suffix == ".pdf" else Docx2txtLoader(tmp_path)
            docs.extend(splitter.split_documents(loader.load()))
        except Exception as e:
            st.warning(f"Erreur {uf.name} : {e}")
        finally:
            os.unlink(tmp_path)
    return docs

def build_chain(vectorstore, llm, profil_key):
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
    qa_prompt = PromptTemplate(input_variables=["context", "chat_history", "question"], template=prompt_template)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=False,
        verbose=False
    )

def init_session():
    for k, v in {"messages": [], "profil": None, "chain": None, "vectorstore": None,
                 "docs_loaded": False, "api_key_ok": False, "preloaded_files": []}.items():
        if k not in st.session_state:
            st.session_state[k] = v

def main():
    init_session()
    st.title("💻 Conseiller Métiers du Numérique")
    st.caption("Un assistant IA pour explorer et choisir votre voie dans le numérique")

    with st.sidebar:
        st.header("⚙️ Configuration")

        api_key = None
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            st.session_state.api_key_ok = True
            st.success("✅ Clé API configurée")
        except:
            api_key = st.text_input("🔑 Clé API Google Gemini", type="password", placeholder="AIza...")
            if api_key:
                st.session_state.api_key_ok = True
                st.success("✅ Clé API configurée")

        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        st.divider()

        # Chargement automatique depuis docs/
        if api_key and not st.session_state.docs_loaded:
            with st.spinner("📚 Chargement des documents..."):
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                    vectorstore, nb_chunks, files = load_docs_from_folder(embeddings)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.docs_loaded = True
                        st.session_state.preloaded_files = files
                except Exception as e:
                    st.warning(f"Docs : {e}")

        st.subheader("📄 Base de connaissance")
        if st.session_state.preloaded_files:
            st.success(f"✅ {len(st.session_state.preloaded_files)} document(s) chargé(s)")
            for f in st.session_state.preloaded_files:
                st.caption(f"📎 {f}")
        else:
            st.info("Aucun document dans le dossier `docs/`")

        with st.expander("➕ Ajouter des documents"):
            uploaded_files = st.file_uploader("PDF / Word supplémentaires", type=["pdf", "docx", "doc"], accept_multiple_files=True)
            if uploaded_files and api_key:
                if st.button("📥 Indexer", use_container_width=True):
                    with st.spinner("Indexation..."):
                        try:
                            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                            new_docs = load_documents_from_upload(uploaded_files)
                            if new_docs:
                                st.session_state.vectorstore = Chroma.from_documents(new_docs, embeddings)
                                st.session_state.docs_loaded = True
                                st.session_state.chain = None
                                st.success(f"✅ {len(new_docs)} extraits indexés !")
                        except Exception as e:
                            st.error(f"Erreur : {e}")

        st.divider()

        st.subheader("👤 Profil détecté")
        if st.session_state.profil:
            p = PROFILS[st.session_state.profil]
            st.markdown(f'<span class="profile-badge {p["badge_class"]}">{p["label"]}</span>', unsafe_allow_html=True)
        else:
            st.info("Auto-détecté à la première question")

        if st.button("🔄 Réinitialiser", use_container_width=True):
            st.session_state.messages = []
            st.session_state.profil = None
            st.session_state.chain = None
            st.rerun()

        st.divider()
        st.caption("POC — Chatbot Métiers du Numérique\nPowered by Gemini Flash 2.0 + RAG")

    if not st.session_state.api_key_ok:
        st.info("👈 Entrez votre clé API Gemini dans le panneau de gauche.")
        st.markdown("Obtenez une clé gratuite sur [aistudio.google.com](https://aistudio.google.com)")
        return

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("""Bonjour ! 👋 Je suis votre **conseiller en métiers du numérique**.

Je suis là pour vous aider à :
- 🎓 Découvrir les métiers du numérique selon votre profil
- 💡 Obtenir des conseils personnalisés d'orientation
- 🚀 Construire votre projet professionnel dans le secteur tech

**Dites-moi qui vous êtes et ce que vous recherchez !**
*(exemple : "Je suis étudiant en terminale et je cherche ma voie dans l'informatique")*""")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Posez votre question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Réflexion en cours..."):
                try:
                    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=api_key)

                    if not st.session_state.profil:
                        profil = detect_profil(user_input, llm)
                        st.session_state.profil = profil
                        p = PROFILS[profil]
                        st.markdown(f'<span class="profile-badge {p["badge_class"]}">Profil détecté : {p["label"]}</span>', unsafe_allow_html=True)

                    if st.session_state.chain is None:
                        if st.session_state.vectorstore:
                            st.session_state.chain = build_chain(st.session_state.vectorstore, llm, st.session_state.profil)
                        else:
                            profil_info = PROFILS[st.session_state.profil]
                            history = "\n".join([f"{m['role'].capitalize()} : {m['content']}" for m in st.session_state.messages[:-1]])
                            full_prompt = f"{profil_info['system']}\n\nHistorique :\n{history}\n\nQuestion : {user_input}\n\nRéponse :"
                            response = llm.invoke(full_prompt)
                            answer = response.content
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            return

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
