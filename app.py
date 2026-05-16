import streamlit as st
import torch
from transformers import pipeline
import time
from datetime import datetime

st.set_page_config(layout="wide", page_title="QA Pro", initial_sidebar_state="expanded")

# CSS personnalisé avec un design moderne et épuré
st.markdown("""
    <style>
    /* Styles généraux */
    .main {
        padding: 2rem;
        background-color: #ffffff; /* Fond blanc pur */
        color: #2c3e50; /* Texte gris foncé */
    }
    
    /* Cartes de métriques */
    .stMetric {
        background-color: #f8f9fa; /* Gris très clair */
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        color: #2c3e50;
        border: 1px solid #e9ecef;
    }
    
    /* Boîte de résultats */
    .result-box {
        background-color: #f0f7ff; /* Bleu très clair */
        padding: 2rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc; /* Bordure bleue discrète */
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Boîte de comparaison */
    .comparison-box {
        background-color: #f8f9fa; /* Gris clair */
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    
    /* Titre principal */
    .header-title {
        color: #1a1a1a;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Sous-titre */
    .header-subtitle {
        color: #666666;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    
    /* Boutons stylisés */
    .stButton>button {
        background-color: #0066cc; /* Bleu moderne */
        color: white;
        border-radius: 0.4rem;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,102,204,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0052a3; /* Bleu plus foncé au survol */
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,102,204,0.3);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    
    /* Badge de statut */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Ligne de séparation */
    hr {
        border: none;
        border-top: 1px solid #e9ecef;
        margin: 1.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal de l'application
st.markdown('<h1 class="header-title">Question-Answering Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="header-subtitle">Système Intelligent d\'Extraction de Réponses avec IA</p>', unsafe_allow_html=True)
st.markdown("---")

# Initialisation de l'état de session
if 'history' not in st.session_state:
    st.session_state.history = []
    
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_questions': 0,
        'avg_confidence': 0,
        'total_time': 0,
        'model_performance': {}
    }

# Barre latérale pour la configuration
with st.sidebar:
    st.header(" Configuration")
    
    models = {
        "DistilBERT (Rapide)": "distilbert-base-uncased-distilled-squad",
        "BERT (Précis)": "deepset/bert-base-cased-squad2",
        "RoBERTa (Puissant)": "deepset/roberta-base-squad2"
    }
    
    model_choice = st.selectbox("Sélectionnez un modèle :", list(models.keys()))
    
    st.markdown("---")
    st.subheader(" Statistiques")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", st.session_state.stats['total_questions'])
    with col2:
        avg_conf = st.session_state.stats['avg_confidence']
        st.metric("Confiance", f"{avg_conf:.1%}")
    
    st.metric("Temps total", f"{st.session_state.stats['total_time']:.2f}s")
    
    st.markdown("---")
    
    if st.button(" Réinitialiser"):
        st.session_state.history = []
        st.session_state.stats = {
            'total_questions': 0,
            'avg_confidence': 0,
            'total_time': 0,
            'model_performance': {}
        }
        st.success("Historique réinitialisé")

# Fonction de mise en cache pour charger le modèle
@st.cache_resource
def load_qa_model(model_name):
    try:
        with st.spinner("Chargement du modèle..."):
            model_id = models[model_name]
            qa = pipeline("question-answering", model=model_id)
        st.success("Modèle chargé avec succès")
        return qa
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        return None

qa_pipeline = load_qa_model(model_choice)

# Onglets de navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Recherche", 
    " Exemples", 
    " Comparaison",
    " Statistiques", 
    " À propos"
])

# ============ ONGLET 1: RECHERCHE ============ 
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Entrez vos données")
        
        context = st.text_area(
            "Contexte :", 
            height=200,
            placeholder="Collez le texte dans lequel vous voulez chercher une réponse...",
            value="Paris est la capitale de la France. Elle est située sur la Seine. La Tour Eiffel est un monument emblématique de Paris. Le Louvre est le musée le plus visité au monde. Notre-Dame est une cathédrale gothique célèbre."
        )
        
        question = st.text_input(
            "Question :", 
            placeholder="Posez votre question...",
            value="Quel est le musée le plus visité au monde ?"
        )
    
    with col2:
        st.subheader("Infos")
        st.metric("Caractères", len(context))
        st.metric("Question", len(question))
        model_name_display = model_choice.split("(")[0].strip()
        st.metric("Modèle", model_name_display)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        search_button = st.button("🔍 Rechercher", use_container_width=True, type="primary")
    with col2:
        clear_button = st.button("🗑️ Effacer", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    if search_button:
        if not context or not question:
            st.warning("Veuillez fournir un contexte ET une question !")
        elif qa_pipeline:
            start_time = time.time()
            
            try:
                with st.spinner("Analyse en cours..."):
                    result = qa_pipeline(question=question, context=context)
                    answer = result['answer']
                    score = result['score']
                    
                elapsed_time = time.time() - start_time
                
                st.session_state.stats['total_questions'] += 1
                st.session_state.stats['total_time'] += elapsed_time
                
                if st.session_state.stats['total_questions'] > 0:
                    st.session_state.stats['avg_confidence'] = (
                        st.session_state.stats['avg_confidence'] * 
                        (st.session_state.stats['total_questions'] - 1) + score
                    ) / st.session_state.stats['total_questions']
                
                st.session_state.history.append({
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'model': model_choice,
                    'question': question,
                    'answer': answer,
                    'confidence': score,
                    'time': elapsed_time
                })
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.success("Réponse trouvée avec succès !")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Réponse", answer)
                with col2:
                    if score > 0.8:
                        badge_class = "status-success"
                        status_text = "Très fiable"
                    elif score > 0.5:
                        badge_class = "status-warning"
                        status_text = "Acceptable"
                    else:
                        badge_class = "status-error"
                        status_text = "À vérifier"
                    
                    st.metric("Confiance", f"{score:.1%}")
                    st.markdown(f'<span class="status-badge {badge_class}">{status_text}</span>', unsafe_allow_html=True)
                with col3:
                    st.metric("Temps", f"{elapsed_time:.2f}s")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.subheader("Localisation dans le texte")
                
                highlighted_context = context.replace(
                    answer, 
                    f"**[{answer}]**"
                )
                st.info(highlighted_context)
                
            except Exception as e:
                st.error(f"Erreur lors de l'analyse: {e}")
        else:
            st.error("Le modèle n'a pas pu être chargé.")

# ============ ONGLET 2: EXEMPLES ============ 
with tab2:
    st.subheader("Exemples Pré-configurés")
    
    examples = {
        "Géographie": {
            "context": "Le Mont Everest est la plus haute montagne du monde avec 8849 mètres. Il est situé dans l'Himalaya entre le Népal et le Tibet. L'Amazonie est la plus grande forêt tropicale du monde.",
            "question": "Quelle est la plus haute montagne du monde ?"
        },
        "Histoire": {
            "context": "La Révolution française a commencé en 1789. Elle a marqué la fin de l'Ancien Régime. Napoléon Bonaparte a émergé comme figure clé après la révolution.",
            "question": "En quelle année la Révolution française a-t-elle commencé ?"
        },
        "Science": {
            "context": "L'eau bout à 100 degrés Celsius au niveau de la mer. La molécule d'eau est composée de deux atomes d'hydrogène et un atome d'oxygène. L'hydrogène est l'élément le plus léger de l'univers.",
            "question": "À quelle température l'eau bout-elle ?"
        }
    }
    
    for category, example in examples.items():
        if st.button(f" {category}", use_container_width=True):
            st.session_state.example_context = example["context"]
            st.session_state.example_question = example["question"]
            st.rerun()

# ============ ONGLET 3: COMPARAISON ============ 
with tab3:
    st.subheader("Comparaison des Modèles")
    st.info("Cette section compare les performances des différents modèles sur les mêmes questions.")
    
    if st.session_state.history:
        st.write("Historique des recherches :")
        for i, entry in enumerate(st.session_state.history):
            with st.expander(f"Recherche {i+1} - {entry['timestamp']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Modèle:** {entry['model']}")
                    st.write(f"**Question:** {entry['question']}")
                with col2:
                    st.write(f"**Réponse:** {entry['answer']}")
                    st.write(f"**Confiance:** {entry['confidence']:.1%}")
                    st.write(f"**Temps:** {entry['time']:.2f}s")
    else:
        st.info("Aucune recherche effectuée pour le moment.")

# ============ ONGLET 4: STATISTIQUES ============ 
with tab4:
    st.subheader("Statistiques Globales")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total questions", st.session_state.stats['total_questions'])
    with col2:
        st.metric("Confiance moyenne", f"{st.session_state.stats['avg_confidence']:.1%}")
    with col3:
        st.metric("Temps total", f"{st.session_state.stats['total_time']:.2f}s")

# ============ ONGLET 5: À PROPOS ============ 
with tab5:
    st.subheader("À propos de cette application")
    st.markdown("""
    ### Question-Answering Pro
    
    Cette application utilise des modèles de deep learning pour extraire automatiquement des réponses à partir d'un texte donné.
    
    **Modèles disponibles:**
    - **DistilBERT**: Modèle rapide et léger, idéal pour les applications en temps réel
    - **BERT**: Modèle précis et équilibré
    - **RoBERTa**: Modèle puissant avec de meilleures performances
    
    **Technologie:**
    - Transformers (Hugging Face)
    - PyTorch
    - Streamlit
    """)
