"""
Semantic Insight Explorer
A production-quality Streamlit app for semantic text similarity analysis.
Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings.
No preprocessing, cleaning, stemming, or model training is performed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import networkx as nx
import io

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Insight Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F8F9FC; }

    /* Hero title */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1A1A2E;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 1.05rem;
        color: #5C5C7B;
        max-width: 680px;
    }
    .hero-badge {
        display: inline-block;
        background: #E8F0FE;
        color: #3D5AFE;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Card containers */
    .card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }

    /* Section headings */
    .section-heading {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1A1A2E;
        margin-bottom: 4px;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    }

    /* Tab font */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        font-weight: 600;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9999BB;
        font-size: 0.8rem;
        padding: 20px 0 8px;
        border-top: 1px solid #E5E7EB;
        margin-top: 32px;
    }

    /* Sidebar heading */
    .sidebar-heading {
        font-size: 1rem;
        font-weight: 700;
        color: #1A1A2E;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the pretrained SentenceTransformer model once and cache it."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ─────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────
def get_embeddings(model, texts: list[str]) -> np.ndarray:
    """Generate sentence embeddings for a list of texts."""
    return model.encode(texts, show_progress_bar=False)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute the full pairwise cosine similarity matrix."""
    return cosine_similarity(embeddings)


def get_top_pairs(texts: list[str], sim_matrix: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    """Extract the top N most similar unique pairs from the similarity matrix."""
    pairs = []
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "Sentence A": texts[i],
                "Sentence B": texts[j],
                "Similarity Score": round(float(sim_matrix[i][j]), 4),
            })
    df = pd.DataFrame(pairs).sort_values("Similarity Score", ascending=False).head(top_n)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df.reset_index(drop=True)


def build_similarity_df(texts: list[str], sim_matrix: np.ndarray) -> pd.DataFrame:
    """Build a labelled DataFrame of the similarity matrix."""
    return pd.DataFrame(sim_matrix, index=texts, columns=texts).round(4)


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 2D using PCA."""
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(embeddings)


# ─────────────────────────────────────────────
# Plotly Chart Builders
# ─────────────────────────────────────────────
def plot_top_pairs_bar(top_df: pd.DataFrame) -> go.Figure:
    """Interactive bar chart of top similar pairs."""
    labels = [f"{r['Sentence A'][:20]}… ↔ {r['Sentence B'][:20]}…"
              if len(r['Sentence A']) > 20 or len(r['Sentence B']) > 20
              else f"{r['Sentence A']} ↔ {r['Sentence B']}"
              for _, r in top_df.iterrows()]
    fig = go.Figure(go.Bar(
        x=labels,
        y=top_df["Similarity Score"],
        marker=dict(
            color=top_df["Similarity Score"],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Score"),
        ),
        text=top_df["Similarity Score"].apply(lambda s: f"{s:.4f}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Top Similar Pairs — Cosine Similarity",
        xaxis_title="Pair",
        yaxis_title="Similarity Score",
        yaxis_range=[0, 1.05],
        xaxis_tickangle=-35,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=13, color="#1A1A2E"),
        xaxis=dict(tickfont=dict(color="#1A1A2E", size=12)),
        yaxis=dict(tickfont=dict(color="#1A1A2E", size=12)),
        margin=dict(b=140),
    )
    return fig


def plot_heatmap(sim_df: pd.DataFrame) -> go.Figure:
    """Annotated heatmap of the full similarity matrix."""
    labels = [t[:30] + "…" if len(t) > 30 else t for t in sim_df.columns]
    fig = go.Figure(go.Heatmap(
        z=sim_df.values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmin=0, zmax=1,
        text=sim_df.values.round(4),
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="A: %{y}<br>B: %{x}<br>Score: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Cosine Similarity Heatmap",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=12, color="#1A1A2E"),
        xaxis=dict(tickfont=dict(color="#1A1A2E", size=12), tickangle=-40),
        yaxis=dict(tickfont=dict(color="#1A1A2E", size=12)),
        margin=dict(l=160, b=160),
    )
    return fig


def plot_pca_scatter(texts: list[str], coords: np.ndarray) -> go.Figure:
    """2D scatter plot of sentence embeddings reduced via PCA."""
    short = [t[:25] + "…" if len(t) > 25 else t for t in texts]
    fig = go.Figure(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers+text",
        text=short,
        textposition="top center",
        marker=dict(
            size=12,
            color=list(range(len(texts))),
            colorscale="Viridis",
            showscale=False,
            line=dict(width=1, color="white"),
        ),
        hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        textfont=dict(color="#1A1A2E", size=13, family="Inter, sans-serif"),
    ))
    fig.update_layout(
        title="2D Embedding Space (PCA)",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        plot_bgcolor="#F8F9FC",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=12, color="#1A1A2E"),
        xaxis=dict(tickfont=dict(color="#1A1A2E", size=12), title_font=dict(color="#1A1A2E")),
        yaxis=dict(tickfont=dict(color="#1A1A2E", size=12), title_font=dict(color="#1A1A2E")),
    )
    return fig


def plot_network_graph(texts: list[str], sim_matrix: np.ndarray, threshold: float = 0.50) -> go.Figure:
    """
    Network graph where nodes are sentences and edges exist when
    similarity > threshold. Edge width scales with similarity score.
    """
    G = nx.Graph()
    short = [t[:22] + "…" if len(t) > 22 else t for t in texts]

    # Add nodes
    for node in short:
        G.add_node(node)

    # Add edges above threshold
    n = len(texts)
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i][j])
            if score > threshold:
                G.add_edge(short[i], short[j], weight=score)

    pos = nx.spring_layout(G, seed=42, k=1.5)

    # Edge traces
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get("weight", 0.5)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(1, weight * 6), color=f"rgba(61,90,254,{weight:.2f})"),
            hoverinfo="none",
        ))

    # Node trace
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            size=18,
            color="#3D5AFE",
            line=dict(width=2, color="white"),
        ),
        hoverinfo="text",
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=f"Semantic Network Graph (edges where similarity > {threshold})",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#F8F9FC",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=12),
        margin=dict(t=60, b=20, l=20, r=20),
        height=520,
    )
    return fig


# ─────────────────────────────────────────────
# Paul's Critical Thinking Section
# ─────────────────────────────────────────────
def render_critical_thinking(texts: list[str], top_df: pd.DataFrame, sim_df: pd.DataFrame):
    """Auto-generate critical thinking commentary based on results."""
    top_pair = top_df.iloc[0]
    mean_sim = sim_df.values[np.triu_indices_from(sim_df.values, k=1)].mean()

    standards = {
        "🔎 Clarity": (
            f"The user provided **{len(texts)} inputs**: "
            + ", ".join(f'"{t}"' for t in texts[:5])
            + ("…" if len(texts) > 5 else "")
            + ". Each input was compared against every other input to reveal semantic relationships."
        ),
        "✅ Accuracy": (
            "Embeddings were generated using the pretrained model "
            "**sentence-transformers/all-MiniLM-L6-v2**. "
            "No retraining, fine-tuning, or weight updates were performed. "
            "All similarity scores reflect the model's original, unmodified knowledge."
        ),
        "📐 Precision": (
            f"All cosine similarity scores are reported to **4 decimal places**. "
            f"The highest observed similarity is **{top_pair['Similarity Score']:.4f}** "
            f"between \"{top_pair['Sentence A']}\" and \"{top_pair['Sentence B']}\". "
            f"The mean pairwise similarity across all pairs is **{mean_sim:.4f}**."
        ),
        "🔗 Relevance": (
            "Four complementary visualisations support the similarity results: "
            "(1) the **bar chart** ranks pairs by score for quick comparison; "
            "(2) the **heatmap** provides a complete matrix overview; "
            "(3) the **PCA scatter** reveals geometric clustering in embedding space; "
            "(4) the **network graph** highlights high-similarity neighbourhoods above 0.50."
        ),
        "🧠 Logic": (
            "Items with high cosine similarity share closely aligned embedding vectors, "
            "meaning the model places their meanings nearby in its high-dimensional space. "
            "Conceptually related terms (e.g. domain synonyms or co-occurring ideas) "
            "cluster together, while unrelated terms are pushed to distant positions."
        ),
        "⭐ Significance": (
            f"The most significant finding is that **\"{top_pair['Sentence A']}\"** "
            f"and **\"{top_pair['Sentence B']}\"** achieved a similarity of "
            f"**{top_pair['Similarity Score']:.4f}**, indicating strong semantic overlap. "
            "This pair is the most meaningful relationship captured in this analysis."
        ),
        "⚖️ Fairness": (
            "Transformer embeddings capture general semantic patterns from large web corpora. "
            "They may **underperform** in highly specialised domains (e.g. legal, medical, or technical jargon) "
            "and can **miss nuance** such as sarcasm, irony, or negation. "
            "Results should be interpreted as approximations of meaning, not ground truth."
        ),
    }

    st.markdown('<div class="section-heading">Paul\'s Critical Thinking Standards</div>', unsafe_allow_html=True)
    st.caption("Automatically generated commentary based on your analysis results.")
    for title, body in standards.items():
        with st.expander(title, expanded=False):
            st.markdown(body)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-heading">🔍 Semantic Insight Explorer</div>', unsafe_allow_html=True)
        st.markdown("---")

        with st.expander("📦 Model Information", expanded=True):
            st.markdown("""
**Model:** `all-MiniLM-L6-v2`  
**Source:** sentence-transformers  
**Embedding dim:** 384  
**Parameters:** ~22 M  
**Licence:** Apache 2.0  
**Runs locally** — no API calls.
""")

        with st.expander("📖 How to Use", expanded=True):
            st.markdown("""
1. Enter one idea per line in the text area.
2. Ideas can be single words, phrases, or paragraphs.
3. Click **Analyse** to generate results.
4. Explore the tabs for charts, the matrix, and search.
5. Download results as CSV.
""")

        with st.expander("ℹ️ About"):
            st.markdown("""
**Semantic Insight Explorer** compares the *meaning* of text inputs using
transformer-based sentence embeddings and cosine similarity.

Built for a university NLP assignment.  
No preprocessing or model training is performed.
""")

        st.markdown("---")
        st.caption("Powered by 🤗 sentence-transformers")


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
def main():
    render_sidebar()

    # Hero header
    st.markdown("""
<div class="hero-badge">NLP · Semantic Similarity · No Training Required</div>
<div class="hero-title">🔍 Semantic Insight Explorer</div>
<div class="hero-sub">
  Compare the <em>meaning</em> of multiple ideas, phrases, or paragraphs side-by-side —
  powered by a single pretrained transformer model running entirely on your machine.
</div>
<br>
""", unsafe_allow_html=True)

    # ── Input Section ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Enter Your Inputs</div>', unsafe_allow_html=True)
    st.caption("One item per line. Words, phrases, or full sentences — all are accepted.")

    default_inputs = (
        "Artificial Intelligence\n"
        "Machine Learning\n"
        "Deep Learning\n"
        "Neural Networks\n"
        "Natural Language Processing\n"
        "Cats"
    )
    raw_input = st.text_area(
        label="inputs",
        value=default_inputs,
        height=180,
        label_visibility="collapsed",
        placeholder="Type one idea per line…",
    )

    col_btn, col_thresh = st.columns([1, 2])
    with col_btn:
        analyse = st.button("🚀 Analyse", use_container_width=True, type="primary")
    with col_thresh:
        edge_threshold = st.slider(
            "Network graph edge threshold",
            min_value=0.10, max_value=0.95, value=0.50, step=0.05,
            help="Edges are drawn only when similarity exceeds this value.",
        )

    if not analyse:
        st.info("Enter your inputs above and click **Analyse** to begin.")
        return

    # ── Validation ─────────────────────────────────────────────────────────────
    texts = [line.strip() for line in raw_input.splitlines() if line.strip()]

    if len(texts) == 0:
        st.error("⚠️ No input detected. Please enter at least two items.")
        return
    if len(texts) == 1:
        st.warning("⚠️ Only one item found. Please enter at least two items to compare.")
        return

    # ── Model & Embeddings ─────────────────────────────────────────────────────
    with st.spinner("Loading model and computing embeddings…"):
        model = load_model()
        embeddings = get_embeddings(model, texts)
        sim_matrix = compute_similarity_matrix(embeddings)
        sim_df = build_similarity_df(texts, sim_matrix)
        top_df = get_top_pairs(texts, sim_matrix, top_n=10)
        coords_2d = pca_2d(embeddings)

    # ── Summary Metrics ────────────────────────────────────────────────────────
    st.markdown("---")
    upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📝 Inputs", len(texts))
    c2.metric("🔗 Pairs Compared", len(upper))
    c3.metric("⬆️ Highest Similarity", f"{upper.max():.4f}")
    c4.metric("📊 Mean Similarity", f"{upper.mean():.4f}")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Charts",
        "🗺️ Heatmap",
        "🌐 Network",
        "🔵 Embedding Space",
        "📋 Similarity Matrix",
        "🏆 Top Pairs",
        "🔎 Search",
    ])

    # ── Tab 1 : Bar Chart ──────────────────────────────────────────────────────
    with tabs[0]:
        st.plotly_chart(plot_top_pairs_bar(top_df), use_container_width=True)

    # ── Tab 2 : Heatmap ────────────────────────────────────────────────────────
    with tabs[1]:
        st.plotly_chart(plot_heatmap(sim_df), use_container_width=True)

    # ── Tab 3 : Network ────────────────────────────────────────────────────────
    with tabs[2]:
        st.plotly_chart(
            plot_network_graph(texts, sim_matrix, threshold=edge_threshold),
            use_container_width=True,
        )
        st.caption(
            f"Edges are drawn when similarity > **{edge_threshold}**. "
            "Adjust the threshold in the control panel above."
        )

    # ── Tab 4 : PCA Scatter ────────────────────────────────────────────────────
    with tabs[3]:
        st.plotly_chart(plot_pca_scatter(texts, coords_2d), use_container_width=True)
        st.caption(
            "Positions are determined by PCA on 384-dimensional embeddings. "
            "Semantically similar items cluster closer together."
        )

    # ── Tab 5 : Similarity Matrix ──────────────────────────────────────────────
    with tabs[4]:
        st.markdown('<div class="section-heading">Full Cosine Similarity Matrix</div>', unsafe_allow_html=True)
        st.dataframe(
            sim_df.style.background_gradient(cmap="Blues", axis=None).format("{:.4f}"),
            use_container_width=True,
        )

    # ── Tab 6 : Top Pairs ─────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown('<div class="section-heading">Top Similar Pairs</div>', unsafe_allow_html=True)
        st.dataframe(top_df, use_container_width=True, hide_index=True)

        # CSV download
        csv_buffer = io.StringIO()
        top_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="semantic_similarity_results.csv",
            mime="text/csv",
        )

    # ── Tab 7 : Search ────────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown('<div class="section-heading">Find Similar Inputs</div>', unsafe_allow_html=True)
        selected = st.selectbox("Select a sentence to find its closest matches:", texts)
        idx = texts.index(selected)
        scores = [(texts[j], round(float(sim_matrix[idx][j]), 4))
                  for j in range(len(texts)) if j != idx]
        scores.sort(key=lambda x: x[1], reverse=True)
        top5 = scores[:5]

        st.markdown(f"**Top 5 matches for:** _{selected}_")
        for rank, (sentence, score) in enumerate(top5, start=1):
            col_a, col_b = st.columns([4, 1])
            col_a.markdown(f"**{rank}.** {sentence}")
            col_b.metric("", f"{score:.4f}")

    # ── Critical Thinking ─────────────────────────────────────────────────────
    st.markdown("---")
    render_critical_thinking(texts, top_df, sim_df)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="footer">
  Semantic Insight Explorer · Built with Streamlit & sentence-transformers ·
  Model: all-MiniLM-L6-v2 · No data leaves your machine
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
