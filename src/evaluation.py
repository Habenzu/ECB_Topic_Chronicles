from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def topic_quality(
    model,
    documents: List[Any],
    embeddings: np.ndarray,
    top_n_words: int = 10,
    sample_size: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute a suite of quality metrics for a fitted BERTopic model.

    This function evaluates topic coherence, diversity, topic size statistics, inter-topic and intra-cluster similarities,
    and clustering metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) using the provided documents and embeddings.
    The steps include tokenizing documents, extracting top words per topic, calculating coherence and diversity,
    analyzing topic size distribution, computing centroid similarities, and evaluating clustering quality.
    Returns a dictionary with all computed metrics for further analysis or comparison.

    | Metric                | What it measures                                                  | "Good" direction  |
    | --------------------- | ----------------------------------------------------------------- | ----------------- |
    | `coherence_umass`     | Top-word co-occurrence within topics (0 >> coherent, -inf worst)  | ↑ towards 0       |
    | `diversity`           | Fraction of unique words across all top-N lists (0-1)             | ↑ higher          |
    | `size_std`            | Standard deviation of topic sizes                                 | ↓ lower           |
    | `size_entropy`        | Evenness of topic sizes (Shannon entropy)                         | ↑ higher          |
    | `mean_inter_topic`    | Average cosine sim between topic centroids                        | ↓ lower           |
    | `silhouette_cosine`   | How well each doc fits its own topic (-1…+1)                      | ↑ higher          |
    | `davies_bouldin`      | Ratio of intra- vs. inter-cluster scatter                         | ↓ lower           |
    | `calinski_harabasz`   | Variance-ratio score                                              | ↑ higher          |
    | `mean_intra_cluster`  | Avg. cosine sim *within* topics                                   | ↑ higher          |
    
    """

    # 1. Tokenise documents exactly as BERTopic did ------------------------
    tokenizer = model.vectorizer_model.build_tokenizer()
    tokenised_docs = [tokenizer(doc) for doc in documents]

    # 2. Top-N word lists per current topic (skip outlier -1) --------------
    topic_words, topic_ids = [], []
    for tid, word_stats in model.get_topics().items():
        if tid == -1 or not word_stats:
            continue
        words = [w for w, _ in word_stats[:top_n_words]]
        if words:
            topic_words.append(words)
            topic_ids.append(tid)

    # 3. Coherence (u_mass) and diversity ----------------------------------
    dictionary = Dictionary(tokenised_docs)
    dictionary.add_documents(topic_words)
    coherence = float(
        CoherenceModel(
            topics=topic_words,
            texts=tokenised_docs,
            dictionary=dictionary,
            coherence="u_mass",
        ).get_coherence()
    )
    diversity = len({w for tw in topic_words for w in tw}) / (len(topic_words) * top_n_words)

    # 4. Topic-size stats ----------------------------------------------------
    doc_labels, _ = model.transform(documents, embeddings=embeddings)
    doc_labels = np.asarray(doc_labels)
    doc_mask = doc_labels != -1
    filtered = doc_labels[doc_mask]
    if filtered.size:
        _, counts = np.unique(filtered, return_counts=True)
        sizes = counts.astype(float)
    else:
        sizes = np.array([], float)

    size_std = float(sizes.std(ddof=0)) if sizes.size else None
    size_entropy = float(
        -(sizes / sizes.sum() * np.log(sizes / sizes.sum())).sum()
    ) if sizes.size else None

    # 5. Mean inter-topic cosine (centroid embeddings) ----------------------
    live_ids = sorted({v for v in model.topic_mapper_.get_mappings().values() if v != -1})
    centroids = model.topic_embeddings_[live_ids]
    inter = cosine_similarity(centroids)
    np.fill_diagonal(inter, 0.0)
    mean_inter_topic = float(inter.mean()) if inter.size else None

    # 6. Cluster metrics on document embeddings -----------------------------
    per_cluster: Dict[int, Dict[str, Any]] = {}
    silhouette = davies = calinski = None

    if doc_mask.sum() > 1 and np.unique(doc_labels[doc_mask]).size > 1:
        X, y = embeddings[doc_mask], doc_labels[doc_mask]

        # optional speed-up
        if sample_size and len(X) > sample_size:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(X), sample_size, replace=False)
            X, y = X[idx], y[idx]

        X = normalize(X)  # for cosine-based measures

        sil_samples = silhouette_samples(X, y, metric="cosine")
        silhouette = float(sil_samples.mean())
        davies = float(davies_bouldin_score(X, y))
        calinski = float(calinski_harabasz_score(X, y))

        for tid in np.unique(y):
            member_mask = y == tid
            members = X[member_mask]
            n = len(members)
            intra = None
            if n > 1:
                sims = cosine_similarity(members)
                intra = float(sims[np.triu_indices(n, 1)].mean())
            per_cluster[int(tid)] = {
                "size": int(n),
                "silhouette": float(sil_samples[member_mask].mean()),
                "intra_cosine": intra,
            }

        intra_values = [d["intra_cosine"] for d in per_cluster.values() if d["intra_cosine"] is not None]
        mean_intra_cluster = float(np.mean(intra_values)) if intra_values else None

    return {
        "coherence_umass": coherence,
        "diversity": diversity,
        "size_std": size_std,
        "size_entropy": size_entropy,
        "mean_inter_topic": mean_inter_topic,
        "silhouette_cosine": silhouette,
        "davies_bouldin": davies,
        "calinski_harabasz": calinski,
        "mean_intra_cluster_cosine": mean_intra_cluster,
        "per_cluster": per_cluster,
    }
