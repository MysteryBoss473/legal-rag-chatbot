# 🤖 Legal RAG Chatbot

> Un chatbot juridique intelligent basé sur le RAG (Retrieval-Augmented Generation) avec l'API Groq et ChromaDB Cloud.

## 📋 Prérequis

- Python 3.10+
- Compte [Groq](https://console.groq.com) avec une clé API
- Compte [Chroma Cloud](https://trychroma.com) avec une base de données

## 🚀 Installation rapide

```bash
# 1. Cloner le repository
git clone https://github.com/MysteryBoss473/legal-rag-chatbot.git
cd legal-rag-chatbot

# 2. Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos clés API

# 5. Indexer les documents PDF
python -m app.indexer

# 6. Lancer l'application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🔧 Configuration

Créez un fichier `.env` à la racine avec les variables suivantes :

| Variable | Description | Exemple |
|----------|-------------|---------|
| `GROQ_API_KEY` | Clé API Groq | `gsk_...` |
| `GROQ_MODEL` | Modèle LLM | `llama-3.3-70b-versatile` |
| `CHROMA_API_KEY` | Clé API Chroma Cloud | `ck_...` |
| `CHROMA_TENANT` | Identifiant tenant | `votre-tenant` |
| `CHROMA_DATABASE` | Nom de la base | `legal-rag-db` |
| `CHUNK_SIZE` | Taille des chunks | `800` |
| `CHUNK_OVERLAP` | Chevauchement | `150` |
| `TOP_K_RETRIEVAL` | Nombre de résultats RAG | `5` |
| `SIMILARITY_THRESHOLD` | Seuil de similarité | `0.35` |

## 📁 Structure du projet

```
legal-rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py              # Point d'entrée FastAPI
│   ├── config.py            # Configuration centralisée
│   ├── indexer.py           # Indexation des PDF
│   ├── rag_engine.py        # Moteur RAG
│   ├── groq_client.py       # Client Groq
│   ├── chroma_client.py     # Client ChromaDB Cloud
│   ├── chunker.py           # Chunking intelligent
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css    # Styles minimalistes
│   │   └── js/
│   │       └── chat.js      # Logique frontend
│   └── templates/
│       └── index.html       # Interface utilisateur
├── scrapping/
│   └── data/                # Dossier contenant les PDFs juridiques
├── tests/
│   └── test_app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── render.yaml
└── README.md
```

## 📄 Indexation des documents

Placez vos fichiers PDF juridiques dans `scrapping/data/`, puis exécutez :

```bash
python -m app.indexer
```

Le système effectue un **chunking intelligent** :
- Découpage par articles, alinéas et paragraphes
- Préservation du contexte juridique (numéros d'articles, sections)
- Chevauchement configuré pour maintenir la cohérence

## 🌐 Déploiement sur Render

1. Connectez votre repository GitHub à [Render](https://render.com)
2. Créez un nouveau **Web Service**
3. Utilisez le fichier `render.yaml` ou configurez manuellement :
   - **Build Command** : `pip install -r requirements.txt && python -m app.indexer`
   - **Start Command** : `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Ajoutez vos variables d'environnement dans l'onglet **Environment**

## 🎯 Fonctionnalités

- ✅ **RAG strict** : Réponses basées uniquement sur les documents indexés
- ✅ **Citations précises** : Sources juridiques citées avec référence exacte
- ✅ **Chunking intelligent** : Découpage adapté aux textes juridiques
- ✅ **Interface sobre** : Design minimaliste et professionnel
- ✅ **Streaming** : Réponses affichées en temps réel
- ✅ **Historique de conversation** : Contexte maintenu dans la session

## 📜 Licence

MIT