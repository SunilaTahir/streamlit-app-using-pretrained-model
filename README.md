# 🔍 Semantic Insight Explorer

A production-quality Streamlit web application for **semantic text similarity analysis** using a single free pretrained NLP embedding model, evaluated against **Paul's Critical Thinking Standards**.

---

## 📌 Overview

Semantic Insight Explorer encodes your texts using `sentence-transformers/all-MiniLM-L6-v2` and computes pairwise cosine similarity — entirely locally, with no paid API and no preprocessing.

The results are presented through:

- A **similarity matrix** (with exact 4-decimal-place scores)
- **Top 10 most similar pairs**
- **Three Plotly graphs** aligned to Paul's Critical Thinking Standards
- An auto-generated **Paul's Standards evaluation** grounded in the actual results
- **Professional notes** referencing real similarity findings

---

## 🚀 Quick Start

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd semantic-insight-explorer
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate.bat      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **First run:** the model (~90 MB) is automatically downloaded from Hugging Face and cached locally. Subsequent runs use the cache.

### 4. Run the application

```bash
streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`.

---

## 📁 Project Structure

```
semantic-insight-explorer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🧠 Model

| Property       | Value                              |
|----------------|------------------------------------|
| **Name**       | `sentence-transformers/all-MiniLM-L6-v2` |
| **Source**     | [Hugging Face 🤗](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| **Dimensions** | 384                                |
| **Parameters** | ~22 M                              |
| **License**    | Apache 2.0                         |

The model is loaded once via `st.cache_resource` and reused across all interactions.

---

## 📊 Features

### User Input
- Enter any number of words, phrases, sentences, or short paragraphs
- One text per line
- Blank lines are automatically ignored

### Output Tabs
| Tab | Contents |
|-----|----------|
| 🗂 Similarity Matrix | Full cosine similarity matrix · 4 decimal places · CSV download |
| 🏆 Top Pairs | Top 10 most similar pairs · CSV download |
| 📈 Graphs | Three Plotly graphs (see below) |
| 🧠 Paul's Standards | Expandable evaluation for all seven standards |
| 📝 Professional Notes | Result-specific notes for each standard |

### Graphs

#### Graph 1 — Top Similarity Scores (Bar Chart)
Horizontal bar chart of the most similar pairs.
**Supports:** Precision & Significance

#### Graph 2 — Similarity Heatmap
Full pairwise cosine similarity matrix visualised as a heatmap.
**Supports:** Accuracy & Relevance

#### Graph 3 — Paul's Critical Thinking Evaluation
Horizontal bar chart showing auto-estimated percentages for all seven Paul standards:
- Clarity · Accuracy · Precision · Relevance · Logic · Significance · Fairness

Scores are **derived from the actual analysis** (input volume, score spread, pair strength, etc.) — not randomly assigned.

---

## 📐 Paul's Critical Thinking Standards

| Standard | What it measures in this app |
|---|---|
| **Clarity** | Sufficient and distinguishable input texts |
| **Accuracy** | Unmodified pretrained model inference |
| **Precision** | Exact 4-decimal scores; score spread |
| **Relevance** | Graphs directly reflect similarity results |
| **Logic** | High-similarity pairs are semantically coherent |
| **Significance** | Strongest pair is clearly identified |
| **Fairness** | Uniform treatment; transformer bias acknowledged |

---

## ⚙️ Technical Notes

- **No preprocessing** — texts are passed verbatim to the model
- **No model training** — weights are frozen
- **No paid API** — everything runs locally
- **Caching** — model loaded once per session via `st.cache_resource`
- **Error handling** — warnings shown for missing or single-text input

---

## 🖥️ System Requirements

| Component | Minimum |
|---|---|
| Python | 3.10 + |
| RAM | 4 GB (8 GB recommended) |
| Disk | ~500 MB (model cache) |
| GPU | Not required (CPU inference) |

---

## 📄 License

This project is released under the **MIT License**.  
The embedded model (`all-MiniLM-L6-v2`) is licensed under **Apache 2.0**.

---

## 🙏 Acknowledgements

- [Sentence Transformers](https://www.sbert.net/) by UKP Lab
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/)
- Richard Paul's Framework for Critical Thinking
