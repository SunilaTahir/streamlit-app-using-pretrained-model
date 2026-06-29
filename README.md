# 🔍 Semantic Insight Explorer

A production-quality Streamlit web application for **semantic text similarity analysis** using a single free pretrained NLP embedding model, evaluated against **Paul's Critical Thinking Standards**.

---

## 📌 Overview

Semantic Insight Explorer encodes your texts using **sentence-transformers/all-MiniLM-L6-v2** and computes pairwise cosine similarity — entirely locally, with no paid API and no preprocessing.

The results are presented through:

- A **similarity matrix** with exact 4-decimal-place scores
- **Top 10 most similar pairs**
- **Three Plotly graphs** aligned to Paul's Critical Thinking Standards
- An auto-generated **Paul's Standards evaluation** grounded in the actual results
- **Professional notes** referencing real similarity findings

---

## 🚀 Quick Start

**Step 1 — Clone or download the project**

Save all three files (app.py, requirements.txt, README.md) into one folder.

**Step 2 — Create a virtual environment (recommended)**

On macOS / Linux, run: python -m venv venv then source venv/bin/activate

On Windows, run: python -m venv venv then venv\Scripts\activate.bat

**Step 3 — Install dependencies**

Run: pip install -r requirements.txt

The model (~90 MB) is automatically downloaded from Hugging Face on first run and cached locally. Subsequent runs use the cache.

**Step 4 — Launch the app**

Run: streamlit run app.py

The browser opens automatically at http://localhost:8501

---

## 📁 Project Structure

The project contains three files:

- app.py — Main Streamlit application
- requirements.txt — Python dependencies
- README.md — This file

---

## 🧠 Model Details

| Property | Value |
|---|---|
| **Name** | sentence-transformers/all-MiniLM-L6-v2 |
| **Source** | Hugging Face |
| **Dimensions** | 384 |
| **Parameters** | ~22 M |
| **License** | Apache 2.0 |

The model is loaded once and reused across all interactions via Streamlit's caching.

---

## 📊 Features

**User Input**
- Enter any number of words, phrases, sentences, or short paragraphs
- One text per line
- Blank lines are automatically ignored

**Output Tabs**

| Tab | Contents |
|---|---|
| 🗂 Similarity Matrix | Full cosine similarity matrix · 4 decimal places · CSV download |
| 🏆 Top Pairs | Top 10 most similar pairs · CSV download |
| 📈 Graphs | Three Plotly graphs (see below) |
| 🧠 Paul's Standards | Expandable evaluation for all seven standards |
| 📝 Professional Notes | Result-specific notes for each standard |

**Graph 1 — Top Similarity Scores**
Horizontal bar chart showing the most similar pairs. Supports Precision and Significance.

**Graph 2 — Similarity Heatmap**
Full pairwise cosine similarity matrix as a colour heatmap. Supports Accuracy and Relevance.

**Graph 3 — Paul's Critical Thinking Evaluation**
Horizontal bar chart showing auto-estimated percentages for all seven standards. Scores are derived from the actual analysis results, not randomly assigned.

---

## 📐 Paul's Critical Thinking Standards

| Standard | What it measures in this app |
|---|---|
| **Clarity** | Sufficient and distinguishable input texts |
| **Accuracy** | Unmodified pretrained model inference |
| **Precision** | Exact 4-decimal scores and score spread |
| **Relevance** | Graphs directly reflect similarity results |
| **Logic** | High-similarity pairs are semantically coherent |
| **Significance** | Strongest pair is clearly identified |
| **Fairness** | Uniform treatment with transformer bias acknowledged |

---

## ⚙️ Technical Notes

- No preprocessing — texts are passed verbatim to the model
- No model training — weights are completely frozen
- No paid API — everything runs locally on your machine
- Caching — model loaded once per session
- Error handling — warnings shown for missing or single-text input

---

## 🖥️ System Requirements

| Component | Minimum |
|---|---|
| Python | 3.10 or higher |
| RAM | 4 GB (8 GB recommended) |
| Disk | ~500 MB for model cache |
| GPU | Not required, CPU inference works fine |

---

## 📄 License

This project is released under the **MIT License**.
The embedded model (all-MiniLM-L6-v2) is licensed under **Apache 2.0**.

---

## 🙏 Acknowledgements

- Sentence Transformers by UKP Lab — https://www.sbert.net
- Streamlit — https://streamlit.io
- Plotly — https://plotly.com
- Richard Paul's Framework for Critical Thinking
