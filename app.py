"""
app.py — RecoSphere Streamlit app

Hybrid (CF + content) recommendations for movies and books, plus a
cross-domain cold-start mode via the shared embedding space.

Run locally: streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.recommend import Recommender

DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

# alpha defaults from src/sweep_alpha.py's empirical results (reports/alpha_sweep_*.json)
DEFAULT_ALPHA = {"movie": 0.25, "book": 1.0}

TITLE_COL_CANDIDATES = ["title", "Title", "book_title", "movie_title", "name", "Name"]


# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RecoSphere",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --rs-bg: #0b0e14;
    --rs-panel: #141925;
    --rs-panel-2: #1b2131;
    --rs-border: #262d40;
    --rs-accent: #7c9eff;
    --rs-accent-2: #ff8fab;
    --rs-accent-3: #4fd1c5;
    --rs-text: #e8ebf3;
    --rs-text-dim: #9aa3b8;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(124,158,255,0.10), transparent 45%),
        radial-gradient(circle at 85% 15%, rgba(255,143,171,0.08), transparent 40%),
        var(--rs-bg);
    color: var(--rs-text);
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: var(--rs-panel);
    border-right: 1px solid var(--rs-border);
}

/* header / hero */
.rs-hero {
    padding: 1.6rem 1.8rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(124,158,255,0.18), rgba(255,143,171,0.10));
    border: 1px solid var(--rs-border);
    margin-bottom: 1.4rem;
}
.rs-hero h1 {
    font-size: 2.1rem;
    margin: 0 0 0.3rem 0;
    background: linear-gradient(90deg, var(--rs-accent), var(--rs-accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.rs-hero p {
    color: var(--rs-text-dim);
    margin: 0;
    font-size: 0.98rem;
}

/* recommendation cards */
.rs-card {
    background: var(--rs-panel-2);
    border: 1px solid var(--rs-border);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.7rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.rs-card:hover {
    transform: translateY(-2px);
    border-color: var(--rs-accent);
}
.rs-card .rs-rank {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--rs-bg);
    background: var(--rs-accent);
    border-radius: 999px;
    padding: 0.1rem 0.55rem;
    margin-right: 0.5rem;
}
.rs-card .rs-title {
    font-size: 1.02rem;
    font-weight: 600;
    color: var(--rs-text);
}
.rs-card .rs-meta {
    color: var(--rs-text-dim);
    font-size: 0.82rem;
    margin-top: 0.2rem;
}
.rs-score-bar-bg {
    width: 100%;
    height: 6px;
    border-radius: 999px;
    background: var(--rs-border);
    margin-top: 0.5rem;
    overflow: hidden;
}
.rs-score-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--rs-accent-3), var(--rs-accent));
}

/* cross-domain card variant */
.rs-card.xdomain {
    border-left: 3px solid var(--rs-accent-2);
}
.rs-card.xdomain .rs-rank {
    background: var(--rs-accent-2);
}

/* section labels */
.rs-section-label {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
    color: var(--rs-text-dim);
    margin: 1.2rem 0 0.5rem 0;
}

/* metric pills */
.rs-pill-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.4rem; }
.rs-pill {
    background: var(--rs-panel);
    border: 1px solid var(--rs-border);
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
    font-size: 0.78rem;
    color: var(--rs-text-dim);
}

/* caveat box */
.rs-caveat {
    background: rgba(255,143,171,0.07);
    border: 1px solid rgba(255,143,171,0.25);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: var(--rs-text-dim);
}

div[data-baseweb="select"] > div {
    background: var(--rs-panel-2) !important;
    border-color: var(--rs-border) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models and embeddings...")
def get_recommender():
    return Recommender()


@st.cache_data(show_spinner=False)
def load_metadata(domain: str) -> pd.DataFrame:
    fname = "movies_clean.parquet" if domain == "movie" else "books_clean.parquet"
    df = pd.read_parquet(DATA_DIR / fname)
    title_col = next((c for c in TITLE_COL_CANDIDATES if c in df.columns), None)
    if title_col is None:
        # last resort: first object/string column that isn't an id
        str_cols = [c for c in df.columns if df[c].dtype == object and "id" not in c.lower()]
        title_col = str_cols[0] if str_cols else df.columns[0]
    df = df.rename(columns={title_col: "_title"})
    id_col = "item_id" if "item_id" in df.columns else df.columns[0]
    df = df.rename(columns={id_col: "_item_id"})
    df["_item_id"] = df["_item_id"].astype(str)
    return df[["_item_id", "_title"]].drop_duplicates(subset="_item_id")


@st.cache_data(show_spinner=False)
def load_alpha_sweep():
    sweeps = {}
    for domain in ["movie", "book"]:
        path = REPORTS_DIR / f"alpha_sweep_{domain}.json"
        if path.exists():
            sweeps[domain] = json.loads(path.read_text())
    return sweeps


def title_for(domain_df: pd.DataFrame, item_id: str) -> str:
    row = domain_df[domain_df._item_id == str(item_id)]
    return row.iloc[0]._title if not row.empty else f"#{item_id}"


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

rec = get_recommender()
sweeps = load_alpha_sweep()

with st.sidebar:
    st.markdown("### 🪐 RecoSphere")
    st.caption("Hybrid CF + content recommender · Movies & Books")

    domain = st.radio("Domain", ["movie", "book"], format_func=lambda d: d.capitalize(), horizontal=True)
    meta_df = load_metadata(domain)

    st.markdown('<div class="rs-section-label">Pick a seed item</div>', unsafe_allow_html=True)
    title_to_id = dict(zip(meta_df._title, meta_df._item_id))
    chosen_title = st.selectbox(
        "Search by title",
        options=sorted(title_to_id.keys()),
        label_visibility="collapsed",
    )
    chosen_item_id = title_to_id[chosen_title]

    st.markdown('<div class="rs-section-label">Tuning</div>', unsafe_allow_html=True)
    default_alpha = DEFAULT_ALPHA.get(domain, 0.5)
    alpha = st.slider(
        "Alpha (CF weight)", 0.0, 1.0, value=default_alpha, step=0.05,
        help="1.0 = pure collaborative filtering, 0.0 = pure content similarity",
    )
    k = st.slider("Number of recommendations", 5, 20, value=10)

    if domain in sweeps:
        best = sweeps[domain]["best_alpha"]
        st.caption(f"Empirically best alpha for {domain}s: **{best}** (from alpha sweep, see reports/)")

    show_cross = st.checkbox("Show cross-domain suggestions", value=True)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="rs-hero">
        <h1>RecoSphere</h1>
        <p>Hybrid recommendations across Movies (MovieLens 100K) and Books (Goodbooks-10k),
        blending collaborative filtering with content embeddings — and a cross-domain
        cold-start mode powered by a shared semantic space.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Main: recs for chosen item
# ---------------------------------------------------------------------------

col_main, col_side = st.columns([2.1, 1])

with col_main:
    st.markdown(f'<div class="rs-section-label">Because you picked</div>', unsafe_allow_html=True)
    st.markdown(f"### {chosen_title}")

    st.markdown(
        f"""
        <div class="rs-pill-row">
            <div class="rs-pill">Domain: {domain}</div>
            <div class="rs-pill">Alpha: {alpha:.2f}</div>
            <div class="rs-pill">Top-{k}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rs-section-label">Recommended for you</div>', unsafe_allow_html=True)

    try:
        results = rec.recommend_in_domain(chosen_item_id, domain=domain, k=k, alpha=alpha)
    except KeyError:
        results = []
        st.warning("This item isn't in the content index — try another title.")

    if results:
        max_score = max(r["score"] for r in results) or 1.0
        min_score = min(r["score"] for r in results)
        span = (max_score - min_score) or 1.0
        for i, r in enumerate(results, start=1):
            rec_title = title_for(meta_df, r["item_id"])
            bar_pct = max(6, int(((r["score"] - min_score) / span) * 100))
            st.markdown(
                f"""
                <div class="rs-card">
                    <span class="rs-rank">#{i}</span>
                    <span class="rs-title">{rec_title}</span>
                    <div class="rs-meta">match score {r['score']:.3f}</div>
                    <div class="rs-score-bar-bg">
                        <div class="rs-score-bar-fill" style="width:{bar_pct}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with col_side:
    st.markdown('<div class="rs-section-label">How this works</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="rs-card" style="margin-bottom:1rem;">
            <div class="rs-meta">
            Score = <b>{alpha:.2f}</b> × CF similarity + <b>{1-alpha:.2f}</b> × content cosine similarity.<br><br>
            CF similarity comes from latent item factors learned via SVD on the ratings matrix.
            Content similarity comes from sentence-transformer embeddings of item metadata.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_cross:
        st.markdown('<div class="rs-section-label">Cross-domain cold-start</div>', unsafe_allow_html=True)
        target_domain = "book" if domain == "movie" else "movie"
        other_meta_df = load_metadata(target_domain)

        try:
            xresults = rec.recommend_cross_domain(chosen_item_id, source_domain=domain,
                                                     target_domain=target_domain, k=5)
        except KeyError:
            xresults = []

        if xresults:
            for i, r in enumerate(xresults, start=1):
                xr_title = title_for(other_meta_df, r["item_id"])
                st.markdown(
                    f"""
                    <div class="rs-card xdomain">
                        <span class="rs-rank">#{i}</span>
                        <span class="rs-title">{xr_title}</span>
                        <div class="rs-meta">{target_domain} · shared-embedding similarity {r['score']:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No cross-domain match found for this item.")

        st.markdown(
            """
            <div class="rs-caveat">
            <b>Honest caveat:</b> cross-domain suggestions rank well below a plain
            popularity baseline in evaluation (see <code>reports/results.md</code>).
            They reflect semantic/topical similarity, not proven taste correlation —
            shown here as an exploratory cold-start path, not a primary recommendation mode.
            </div>
            """,
            unsafe_allow_html=True,
        )
