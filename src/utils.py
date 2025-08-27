import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix()) # path to root of project

import nltk
import spacy
from collections import Counter
from sklearn.feature_extraction import text as sklearn_text
from nltk.corpus import stopwords
import re
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import string
import html
from matplotlib.colors import to_hex
import polars as pl

class StopwordCollector:
    def __init__(self, texts = None, authors = None, min_freq=2500):
        self.texts = texts
        self.authors = authors
        self.min_freq = min_freq
        self._stopword_list = None

    def _load(self):
        nltk.download('stopwords', quiet=True)

        # Additional English stopwords
        additional_english = {
            "there's", "what's", "why's", "where's", "that's", "who's",
            'would', "let's", "here's", 'could', "when's", 'ought', "how's"
        }

        # Very common words from content
        if self.texts is not None: 
            all_words = [
                word
                for doc in self.texts
                for word in doc.lower().split()
            ]
            common_words = {
                word for word, freq in Counter(all_words).items()
                if freq > self.min_freq
            }
        else: 
            common_words = set()
        
        # Author tokens as stopwords
        if self.authors is not None:
            tokens = list(chain.from_iterable(
                re.split(r"\s*(?:,|;| and )\s*", author.strip())
                for author in self.authors if author
            ))
            name_tokens = list(chain.from_iterable(token.split() for token in tokens))
            author_stopwords = {t.lower() for t in name_tokens if t}
        else:
            author_stopwords = set()

        # Built-in stopwords
        sk_en = set(sklearn_text.ENGLISH_STOP_WORDS)
        nl_en = set(stopwords.words('english'))
        sp_de = spacy.load("de_core_news_sm").Defaults.stop_words
        sp_en = spacy.load("en_core_web_sm").Defaults.stop_words

        combined = (
            sk_en |
            nl_en |
            sp_en |
            sp_de |
            common_words |
            additional_english |
            author_stopwords)

        self._stopword_list = sorted(combined)

    def get_stopwords(self):
        if self._stopword_list is None:
            self._load()
        return self._stopword_list

# ------- PLOTTING UTILS -------
def plot_model_metrics(
    data: dict[str, dict[str, float]],
    n_cols: int = 3,
    figsize: tuple[int, int] = (18, 12)):
    """
    Plot every metric in a grid (default 3x3) for an arbitrary number of models.

    Parameters
    ----------
    data       Nested dict:  {model: {metric: value, …}, …}
    n_cols     How many subplot columns (rows are inferred)
    figsize    Overall figure size
    value_fmt  Format string for bar-end labels
    """
    models   = list(data.keys())
    metrics  = list(next(iter(data.values())).keys())
    n_models = len(models)
    n_charts = len(metrics)
    n_rows   = ceil(n_charts / n_cols)

    # get as many distinct colours as models
    cmap   = plt.get_cmap("tab10", n_models)
    colours = [cmap(i) for i in range(n_models)]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for i, metric in enumerate(metrics):
        r, c      = divmod(i, n_cols)
        ax        = axes[r, c]
        values    = [data[m][metric] for m in models]
        x         = np.arange(n_models)
        bars      = ax.bar(x, values, color=colours, width=0.7)

        # bar-end labels
        for bar, val in zip(bars, values):
            offset = 0.02 * max(abs(val), abs(max(values, key=abs)))
            ypos   = val + offset if val >= 0 else val - offset
            va     = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    ypos,
                    format(val, ".4g"),
                    ha="center", va=va, fontsize=8)

        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # turn off any unused sub-axes
    for j in range(n_charts, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    fig.suptitle(f"Model Metrics - {', '.join(models)}", fontsize=16)
    fig.tight_layout()
    plt.close(fig) 

    return fig

def interactive_plot_model_metrics(
    data: dict[str, dict[str, float]],
    n_cols: int = 3,
    figsize_px: tuple[int, int] = (1200, 800),
):
    """
    Interactive grid of bar-plots (one metric per subplot) with a
    model-legend at the bottom.

    *Models* are shown as short labels (A, B, C … or 1, 2, 3 …),
    and the legend links each label to the full model name.

    Parameters
    ----------
    data         {model: {metric: value, …}, …}
    metric_desc  Optional {metric: "long description"}
    n_cols       Number of columns in the subplot grid
    figsize_px   Overall figure size in pixels
    """
    if not data:
        raise ValueError("`data` must not be empty.")

    # 1. Prep names, colours, labels
    models   = list(data.keys())
    metrics  = list(next(iter(data.values())).keys())
    n_rows   = ceil(len(metrics) / n_cols)

    metric_desc_default = {
    "coherence_umass":
        "UMass topic coherence: average log-probability that a pair of top-words "
        "co-occurs in a reference corpus. Closer to 0 (less negative) ⇒ more coherent.",
    "diversity":
        "Topic diversity: proportion of unique top-N words across all topics. "
        "Higher means topics share fewer words and are more distinct.",
    "size_std":
        "Standard deviation of topic sizes (share of documents per topic). "
        "Lower indicates topic sizes are more uniform.",
    "size_entropy":
        "Shannon entropy of the topic-size distribution. "
        "Higher entropy ⇒ topics are more evenly balanced.",
    "mean_inter_topic":
        "Mean cosine similarity between topic embeddings. "
        "Lower values mean topics are further apart (more distinct).",
    "silhouette_cosine":
        "Silhouette score using cosine distance (-1 to 1). "
        "Higher = better cluster separation and cohesion.",
    "davies_bouldin":
        "Davies-Bouldin index (≥ 0). "
        "Lower values indicate tighter, better-separated clusters.",
    "calinski_harabasz":
        "Calinski-Harabasz index (≥ 0). "
        "Higher scores imply well-defined, separated clusters.",
    "mean_intra_cluster_cosine":
        "Mean cosine similarity of documents *within* each cluster/topic. "
        "Higher ⇒ documents in a cluster are more alike.",}
    
    metric_desc = {}
    for metric in metrics:
        if metric in metric_desc_default.keys():
            metric_desc[metric] = metric_desc_default[metric]

    # Short labels: A-Z if ≤26 models, otherwise 1,2,3…
    if len(models) <= 26:
        short_pool = list(string.ascii_uppercase)
    else:
        short_pool = [str(i + 1) for i in range(len(models))]
    model2short = {m: short_pool[i] for i, m in enumerate(models)}

    palette = [to_hex(plt.get_cmap("tab10", len(models))(i)) for i in range(len(models))]
    colour_mapping  = {m: palette[i % len(palette)] for i, m in enumerate(models)}  


    # Titles with hoverable descriptions
    subplot_titles = []
    for metric in metrics:
        label = metric.replace("_", " ").title()
        if metric_desc and metric in metric_desc:
            desc = html.escape(metric_desc[metric], quote=True)
            subplot_titles.append(f'<span title="{desc}">{label}</span>')
        else:
            subplot_titles.append(label)

    # 2. Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.10,
    )

    for m_idx, metric in enumerate(metrics):
        row, col = divmod(m_idx, n_cols)

        # one trace *per model* so each gets a legend entry
        for model in models:
            short = model2short[model]
            value = data[model][metric]

            fig.add_trace(
                go.Bar(
                    x=[short],
                    y=[value],
                    name=f"{short} - {model}",
                    marker=dict(color=colour_mapping[model]),
                    legendgroup=model,
                    showlegend=(m_idx == 0),   # legend only once
                    hovertemplate=(
                        f"{model} ({short})<br>"
                        "%{y:.4g}<extra></extra>"
                    ),
                    offsetgroup=short,
                    width=0.65,
                ),
                row=row + 1,
                col=col + 1,
            )

        fig.update_yaxes(
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(0,0,0,0.25)",
            gridcolor="rgba(0,0,0,0.12)",
            row=row + 1,
            col=col + 1,
        )
        fig.update_xaxes(row=row + 1, col=col + 1)

    # 3. Layout tweaks: legend at bottom, neat margins
    fig.update_layout(
        title=f"Model Metrics - {', '.join(models)}",
        width=figsize_px[0],
        height=figsize_px[1],
        bargap=0.25,
        template="simple_white",
        margin=dict(t=100, b=120, l=40, r=40),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.25,
            xanchor="center", x=0.5,
            title_text="Model Legend",
            font=dict(size=12),
        ),
    )

    # 4. Make subplot titles hoverable  ← NEW
    if metric_desc:
        for ann, metric in zip(fig.layout.annotations, metrics):
            desc = metric_desc.get(metric)
            if desc:
                ann.update(
                    hovertext=desc,
                    hoverlabel=dict(bgcolor="white"),
                )
    return fig

def plot_cluster_metrics(
    metrics: dict[int, dict[str, float]],
    figsize: tuple[int, int] = (10, 9),
    top_n_topics: int = 10,
    model_name: str | None = None,
    id_to_name: dict[int, str] | None = None,   # ← NEW
):
    """
    3-row area chart (size, silhouette, intra-cluster cosine) for the
    *top_n_topics* clusters.

    Parameters
    ----------
    metrics      {cluster_id: {'size': float,
                               'silhouette': float,
                               'intra_cosine': float}, …}
    figsize      Figure size in inches.
    top_n_topics How many clusters to plot (sorted by ID).
    model_name   Optional text for the overall title.
    id_to_name   Optional mapping {cluster_id: 'Topic name', …}.
                 Names are shown on the x-axis; IDs used if missing.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # 1. Prep data
    alpha = 0.6
    clusters = sorted(metrics.keys())[:top_n_topics]

    size_vals       = [metrics[c]['size']        for c in clusters]
    sil_vals        = [metrics[c]['silhouette']  for c in clusters]
    intra_cos_vals  = [metrics[c]['intra_cosine'] for c in clusters]

    # x-positions (0, 1, 2, …) so we can label with names separately
    x = np.arange(len(clusters))

    if id_to_name:
        x_labels = [id_to_name.get(cid, str(cid)) for cid in clusters]
    else:
        x_labels = [str(cid) for cid in clusters]

    # colours
    cmap     = plt.get_cmap('tab10', 3)
    colours  = [cmap(i) for i in range(3)]

    # 2. Plot
    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True,
        figsize=figsize, constrained_layout=True
    )

    title_root = "Topic-based Metrics"
    fig.suptitle(
        f"{title_root} for {model_name}" if model_name else title_root,
        fontsize=16
    )

    # row 1 – size
    axes[0].fill_between(x, size_vals, color='#808080', alpha=alpha)
    axes[0].plot(x, size_vals, 'o', ms=2, color='#808080', zorder=3)
    axes[0].set_title("Topic Cluster Size")
    axes[0].set_ylabel("Size")
    axes[0].grid(alpha=0.3)

    # row 2 – silhouette
    axes[1].fill_between(x, sil_vals, color=colours[1], alpha=alpha)
    axes[1].plot(x, sil_vals, 'o', ms=2, color=colours[1], zorder=3)
    axes[1].set_title("Silhouette Score")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(alpha=0.3)

    # row 3 – intra-cluster cosine
    axes[2].fill_between(x, intra_cos_vals, color=colours[2], alpha=alpha)
    axes[2].plot(x, intra_cos_vals, 'o', ms=2, color=colours[2], zorder=3)
    axes[2].set_title("Intra-topic Cosine")
    axes[2].set_xlabel("Topic")
    axes[2].set_ylabel("Cosine Similarity")
    axes[2].grid(alpha=0.3)

    # shared x-axis ticks / labels & vertical dotted grid
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=35, ha='right')
        ax.tick_params(axis='x', which='both', length=0)
        ax.grid(axis='x', which='minor', linestyle=':', alpha=0.3)

    fig.tight_layout()
    plt.close(fig)
    return fig

def plot_topic_trends(
    df: pl.DataFrame,
    date_col: str = "date",
    topic_col: str = "dominant_topic",
    initial_k: int = 5,
    ignore_topic:int = -1,
    topics_to_show: list = None,
) -> go.Figure:
    """
    Date columns must be datetime polars type, topic column must be string or int.
    Interactive topic-trend chart with three controls:

    • *Slider* How many of the most-frequent topics are visible  
    • *Freq.* Toggle relative (%) vs absolute document counts  
    • *Period* Re-aggregate on the fly: none, 1 w, 2 w, 1 m, 2 m, 3 m, 6 m, 1 y
    """
    agg_map = {
        "No Agg": None,
        "1 Week": "1w",
        "2 Weeks": "2w",
        "1 Month": "1mo",
        "2 Months": "2mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
    }
    if topics_to_show is not None:
        df = df.filter(pl.col(topic_col).is_in(topics_to_show))
    if ignore_topic is not None:
        df = df.filter(pl.col(topic_col) != ignore_topic)
    topics = [t for t, _ in Counter(df[topic_col].to_list()).most_common()]
    store = {}
    for lbl, dur in agg_map.items():
        col = "period"
        if dur is None:  # ---- FIX: cast at top level, not in .dt
            dfa = df.with_columns(pl.col(date_col).cast(pl.Date).alias(col))
        else:
            dfa = df.with_columns(pl.col(date_col).dt.truncate(dur).alias(col))

        tot = dfa.group_by(col).agg(pl.count().alias("tot"))
        tmp = (
            dfa.group_by([col, topic_col])
            .agg(pl.count().alias("cnt"))
            .join(tot, on=col)
            .with_columns((pl.col("cnt") / pl.col("tot") * 100).alias("pct"))
        )

        dates = tmp[col].unique().sort().to_list()
        y_abs = {t: [0] * len(dates) for t in topics}
        y_pct = {t: [0] * len(dates) for t in topics}
        for d, tp, c, _, p in tmp.iter_rows():
            idx = dates.index(d)
            y_abs[tp][idx] = c
            y_pct[tp][idx] = p

        store[lbl] = {
            "x": dates,
            "abs": [y_abs[t] for t in topics],
            "rel": [y_pct[t] for t in topics],
        }

    base = store["1 Month"]
    fig = go.Figure(
        [
            go.Scatter(
                x=base["x"],
                y=base["rel"][i],
                mode="lines",
                name=f"Topic {t}",
            )
            for i, t in enumerate(topics)
        ]
    )

    vis = [[i < k for i in range(len(topics))] for k in range(1, len(topics) + 1)]
    fig.update_layout(
        title="Topic Trends Over Time",
        xaxis_title="Period",
        yaxis_title="Relative Frequency (%)",
        legend_title="Topic",
        sliders=[
            {
                "active": initial_k - 1,
                "steps": [
                    {"label": str(k), "method": "update", "args": [{"visible": v}]}
                    for k, v in enumerate(vis, 1)
                ],
            }
        ],
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Relative (%)",
                        "method": "update",
                        "args": [
                            {"y": base["rel"]},
                            {"yaxis": {"title": "Relative Frequency (%)"}},
                        ],
                    },
                    {
                        "label": "Absolute",
                        "method": "update",
                        "args": [
                            {"y": base["abs"]},
                            {"yaxis": {"title": "Document Count"}},
                        ],
                    },
                ],
                "direction": "down",
                "x": 1.0,
                "xanchor": "right",
                "y": 1.15,
                "yanchor": "top",
                "showactive": True,
            },
            {
                "buttons": [
                    {
                        "label": lbl,
                        "method": "update",
                        "args": [
                            {"x": [itm["x"]] * len(topics), "y": itm["rel"]},
                            {"yaxis": {"title": "Relative Frequency (%)"}},
                        ],
                    }
                    for lbl, itm in store.items()
                ],
                "direction": "down",
                "x": 0.85,
                "xanchor": "right",
                "y": 1.15,
                "yanchor": "top",
                "showactive": True,
            },
        ],
    )

    if topics_to_show is not None: 
        for i, flag in enumerate(vis[len(topics_to_show) - 1]):
            fig.data[i].visible = flag
    else: 
        for i, flag in enumerate(vis[initial_k - 1]):
            fig.data[i].visible = flag

    return fig