#  Fine-Tuned Question-Answering Application

##  Description

Application Streamlit pour l'extraction automatique de réponses à partir d'un contexte donné, utilisant des modèles BERT fine-tunés sur le dataset SQuAD v2.

##  Objectifs du Projet

-  Fine-tuner 3 modèles BERT (DistilBERT, ALBERT, RoBERTa) sur SQuAD v2
-  Comparer leurs performances (F1-score, Exact Match, temps d'inférence)
-  Créer une interface web interactive avec Streamlit
-  Déployer l'application sur Hugging Face Spaces

##  Architecture

### Modèles Utilisés
**DistilBERT** : Un modèle léger et rapide avec 66 millions de paramètres. C'est une version compressée de BERT qui offre un excellent compromis entre vitesse et précision. Idéal pour les applications en temps réel.

**RoBERTa** : Une version améliorée de BERT avec 125 millions de paramètres. Il a été entraîné avec une meilleure méthodologie et offre généralement les meilleures performances, mais au prix d'une vitesse d'inférence plus lente.

**ALBERT** : Un modèle très léger avec seulement 12 millions de paramètres. C'est le plus rapide mais aussi le moins précis des quatre modèles testés.

### Dataset

- **SQuAD v2** : 100k+ questions avec réponses extraites du contexte
- **Caractéristiques** : Questions adversariales, réponses multiples, contextes variés

## Résultats

### Tableau Comparatif
Les résultats de l'évaluation montrent des différences intéressantes entre les modèles :

**DistilBERT** obtient un F1-score de 17.55% avec un Exact Match de 8.7% et un temps d'inférence très rapide de 170 ms. C'est un excellent choix si tu as besoin de vitesse.

**RoBERTa** atteint le meilleur F1-score de 27.77% avec un Exact Match de 11.5% et un temps d'inférence de 331 ms. C'est le meilleur choix si la précision est ta priorité.

**ALBERT** offre un F1-score de 20.83% avec un Exact Match de 10.0% et un temps d'inférence de 351 ms. C'est un bon compromis entre les deux autres modèles.

##  Installation

### Prérequis

- Python 3.8+
- pip ou conda

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/Massilia2001/Projet-QA.git
cd Projet-QA

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

 Utilisation
Lancer l'Application Streamlit
streamlit run app.py

Puis ouvrez http://localhost:8501 dans ton navigateur.

## Fonctionnalités
 Recherche : Pose une question et obtiens une réponse instantanée
 Exemples : Teste des exemples pré-configurés
 Comparaison : Compare les résultats de différents modèles
 Statistiques : Visualise les performances globales
 À propos : En savoir plus sur le projet

 Notebooks
Le dossier notebooks/ contient les notebooks Jupyter pour le fine-tuning et l'évaluation :

01_finetuning_distilbert.ipynb : Fine-tuning de DistilBERT sur SQuAD v2
02_finetuning_bert.ipynb : Fine-tuning de BERT sur SQuAD v2
03_finetuning_roberta.ipynb : Fine-tuning de RoBERTa sur SQuAD v2
04_evaluation_comparison.ipynb : Évaluation et comparaison des modèles
 Déploiement
L'application est déployée sur Hugging Face Spaces :

 Lien : https://huggingface.co/spaces/Massilia2001/fine-tuned-qa-app

 Structure du Projet
Projet-QA/
├── app.py                          # Application Streamlit
├── requirements.txt                # Dépendances Python
├── README.md                       # Ce fichier
├── .gitignore                      # Fichiers à ignorer
├── notebooks/
│   ├── 01_finetuning_distilbert.ipynb
│   ├── 02_finetuning_bert.ipynb
│   ├── 03_finetuning_roberta.ipynb
│   └── 04_evaluation_comparison.ipynb
├── results/
│   ├── comparison_metrics.png
│   ├── radar_comparison.png
│   └── size_vs_performance.png
└── scripts/
    ├── 03_finetune_bert.py
    ├── 05_finetune_roberta.py
    ├── 06_finetune_distilbert.py
    └── 07_Albert.py

 Technologies Utilisées
Streamlit : Interface web interactive
Transformers (Hugging Face) : Modèles BERT pré-entraînés
PyTorch : Framework de deep learning
Datasets : Chargement de SQuAD v2
Pandas & Matplotlib : Analyse et visualisation des données
 Méthodologie
Fine-tuning
Chargement du dataset : SQuAD v2 (100k+ exemples)
Prétraitement : Tokenization, padding, truncation
Configuration : Hyperparamètres optimisés pour CPU
Entraînement : 1-3 epochs avec validation
Évaluation : Calcul des métriques (F1, EM, etc.)
Sauvegarde : Modèles fine-tunés sauvegardés localement
Hyperparamètres
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 3e-5
MAX_LENGTH = 384
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

 Résultats et Analyse
Observations Clés
DistilBERT : Meilleur compromis vitesse/précision
RoBERTa : Meilleure précision mais plus lent
ALBERT : Très rapide mais moins précis
Défis Rencontrés
 Limitations de RAM sur CPU
 Temps d'entraînement long
 Gestion des fichiers volumineux sur GitHub
Solutions Apportées
 Réduction du dataset pour CPU
 Utilisation de batch size réduit
 Stockage des modèles sur Hugging Face Hub
 

 Auteurs
Massilia Oumaza , Mouhammed Bachir Mbye

GitHub : @Massilia2001
Hugging Face : @Massilia2001
.

Dernière mise à jour : 16 mai 2026


