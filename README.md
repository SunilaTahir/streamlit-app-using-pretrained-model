# 🔍 Semantic Insight Explorer

> A production-quality NLP web application that compares the *meaning* of multiple ideas, phrases, or sentences using a single pretrained transformer model — no training, no APIs, no preprocessing.

---

## 📌 Project Overview

**Semantic Insight Explorer** is a Streamlit application built for a university NLP assignment. Instead of comparing just two sentences, it allows users to input any number of items (words, phrases, or full sentences) and performs an exhaustive pairwise semantic similarity analysis using cosine similarity on transformer embeddings.

**Model used:** `sentence-transformers/all-MiniLM-L6-v2`  
**Embedding dimension:** 384  
**Parameters:** ~22 M  
**Licence:** Apache 2.0  

No OpenAI API, Gemini API, HuggingFace Inference API, or any paid service is used. Everything runs locally.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Multi-input comparison** | Enter as many words, phrases, or sentences as needed |
| **Similarity matrix** | Full N×N cosine similarity table with colour gradient |
| **Top 10 pairs** | Ranked list of the most similar pairs with exact 4 d.p. scores |
| **Bar chart** | Interactive Plotly bar chart of top similar pairs |
| **Heatmap** | Annotated Plotly heatmap of the complete similarity matrix |
| **PCA scatter** | 2D projection of 384-dim embeddings for visual clustering |
| **Network graph** | Semantic network where edges appear when similarity > threshold |
| **Search** | Select any sentence and find its top 5 closest matches |
| **CSV download** | Export results to CSV with one click |
| **Critical thinking** | Auto-generated commentary covering all 7 Paul's standards |
| **Cached model** | `st.cache_resource` ensures the model loads only once |
| **Error handling** | Informative warnings for empty input or single-sentence input |

---

## 🛠 Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/semantic-insight-explorer.git
cd semantic-insight-explorer

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push the project to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**.
4. Select your repository, branch (`main`), and set the main file path to `app.py`.
5. Click **Deploy**.

Streamlit Community Cloud will automatically:
- Install all packages from `requirements.txt`
- Download the model from HuggingFace on first run (cached for subsequent runs)
- Serve the app on a public URL

> **Note:** The first cold start may take 1–2 minutes while the model (~90 MB) is downloaded. Subsequent loads are fast thanks to `st.cache_resource`.

---

## 📂 Project Structure

```
semantic-insight-explorer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---


## 🔭 Future Improvements

- **More models:** Allow users to switch between multiple sentence-transformer models (e.g. `all-mpnet-base-v2`, `multi-qa-MiniLM-L6-cos-v1`) and compare their outputs.
- **3D PCA / UMAP:** Extend the embedding visualisation to 3D or use UMAP for more faithful topology preservation.
- **File upload:** Accept `.txt` or `.csv` files as input so users can analyse large corpora.
- **Clustering:** Add automatic K-Means or DBSCAN clustering on the embeddings and colour-code the PCA/network graphs.
- **Multilingual support:** Switch to `paraphrase-multilingual-MiniLM-L12-v2` to handle non-English inputs.
- **Sentence highlighting:** Highlight specific keywords that drive high similarity scores.
- **Threshold animation:** Animate the network graph as the similarity threshold sweeps from 0 to 1.

---

## 📄 Licence

This project is released for educational purposes under the MIT Licence.
