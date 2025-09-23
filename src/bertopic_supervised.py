# pip install sentence-transformers scikit-learn bertopic
import numpy as np
import joblib

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
import polars as pl

from scipy.sparse import csr_matrix, vstack
from catboost import CatBoostClassifier
import numpy as np
from loguru import logger
from pathlib import Path
from tqdm import tqdm

from .utils import get_mapping
from .evaluation import compute_all_metrics, plot_score_gt_pred

FILE_DIR = Path(__file__).parent

def data_loading(path:Path, prototype = False):
    logger.info(f"Loading embedding dataset from {path.as_posix()}")
    df = pl.read_parquet(path)
    if prototype: 
        df = df.sample(100)

    needed_columns = ["id", "topic_label", "chunks", "chunks_embeddings"]
    if not set(needed_columns).issubset(df.columns):
        raise ValueError(f"Missing columns in {path}: {[x for x in needed_columns if x not in df.columns]}")

    # Explode each chunk/embedding into an own row
    df = (
        df.explode(["chunks", "chunks_embeddings"])
        .with_columns(
            pl.arange(0, pl.len()).over("id").alias("chunk_id")
        ))

    # Loading train, validation and test instances
    train_set  = set(pl.read_parquet(path.parent / "train_topics.parquet")['id'].to_list())    
    val_set    = set(pl.read_parquet(path.parent / "val_topics.parquet")['id'].to_list())
    test_set   = set(pl.read_parquet(path.parent / "test_topics.parquet")['id'].to_list())
    train_df = df.filter(pl.col("id").is_in(train_set))
    val_df   = df.filter(pl.col("id").is_in(val_set))
    test_df  = df.filter(pl.col("id").is_in(test_set))

    return (train_df, val_df, test_df)

def binarize_y(train_df, val_df, test_df, n_topics: int = 102):
    logger.info("Binarizing the labels.")
    y_all_multi = [[x] for x in list(range(n_topics))] # All possible topic labels, needed for binarizer
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(y_all_multi)
    y_train_bin = mlb.transform(train_df['topic_label'].to_list()) 
    y_val_bin = mlb.transform(val_df['topic_label'].to_list()) 
    y_test_bin = mlb.transform(test_df['topic_label'].to_list()) 

    return mlb, (y_train_bin, y_val_bin, y_test_bin)

def tune_thresholds(y_true, y_prob, grid=np.linspace(0.05, 0.8, 16)):
    logger.info("Started tuning the tresholds.")
    best = np.zeros(y_true.shape[1])
    for j in range(y_true.shape[1]):
        yj = y_true[:, j]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (y_prob[:, j] >= t).astype(int)
            f1 = f1_score(yj, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best[j] = best_t
    return best

def create_term_representations(docs_train:pl.DataFrame, y_train, n_topics, top_k = 15, label_names = True): 
    logger.info("Creating term representations per topic.")
    # 1. Fit vectorizer on TRAIN ONLY
    vectorizer = CountVectorizer(min_df=5, ngram_range=(1, 2), stop_words="english")
    X_counts = vectorizer.fit_transform(docs_train)
    terms = vectorizer.get_feature_names_out()
    # 2. Build class-document matrix (sum doc term-counts per label) — keep it sparse
    rows = []
    has_docs = []
    for j in range(n_topics):
        idx = y_train[:, j].astype(bool)
        if idx.sum() == 0:
            # placeholder empty sparse row
            rows.append(csr_matrix((1, X_counts.shape[1]), dtype=np.float64))
            has_docs.append(False)
        else:
            # Sum selected rows; .sum(axis=0) returns 1xV matrix
            summed = X_counts[idx].sum(axis=0)
            rows.append(csr_matrix(summed))
            has_docs.append(True)

    label_term_counts = vstack(rows)  # csr: (n_labels, n_terms)

    # 3) Class TF-IDF (ctfidf)
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)
    W = ctfidf.fit_transform(label_term_counts)

    # 4) Extract top-k terms per label efficiently from sparse rows
    mapping = get_mapping(id_to_label=True)
    label_to_top_terms = {}

    for j in range(n_topics):
        row = W.getrow(j)  # csr 1xV
        if row.nnz == 0 or not has_docs[j]:
            label_to_top_terms[j] = []
            continue

        # Get indices of top-k by row.data (already only nonzeros)
        data = row.data
        idxs = row.indices
        if data.size <= top_k:
            top_indices = idxs[np.argsort(-data)]
        else:
            # partial sort for speed; then sort those descending
            part = np.argpartition(-data, top_k - 1)[:top_k]
            top_indices = idxs[part[np.argsort(-data[part])]]

        if label_names: 
            label_to_top_terms[mapping.get(j)] = [terms[i] for i in top_indices]
        else:
            label_to_top_terms[j] = [terms[i] for i in top_indices]

    return label_to_top_terms

def train(X_train, y_train,X_val, y_val, **cat_boost_kwargs):
    """
    1. Training the classifier
    2. Tuning the tresholds on the validation dataset
    3. Creating term representations (similar to BERTopic implementation)
    """
    logger.info("Starting training the cat_boost classifier.")
    logger.info(f"Parameters: {cat_boost_kwargs}")
    base = CatBoostClassifier(**cat_boost_kwargs)
    clf = OneVsRestClassifier(base, n_jobs=1, verbose = 5)
    clf.fit(X_train, y_train)

    logger.info("Tuning the tresholds on the validation dataset.")
    probs_val = clf.predict_proba(X_val)
    thr = tune_thresholds(y_val, probs_val)

    model_data = {
        "base": base, 
        "clf": clf, 
        "thr": thr}

    return model_data

def predict(embeddings, clf, mlb, thr=None, ids=None, agg=None):
    X = np.asarray(embeddings)
    n_labels = len(mlb.classes_)
    thr = thr if thr is not None else np.full(n_labels, 0.5)
    prob = clf.predict_proba(X)

    if agg is None:  # chunk-level
        pred_bin = (prob >= thr).astype(int)
    else:  # doc-level
        ids = np.asarray(ids)
        uniq, inv = np.unique(ids, return_inverse=True)
        if agg == "max":
            prob_doc = np.zeros((len(uniq), n_labels)) 
            counts = np.zeros(len(uniq))
            for i, p in zip(inv, prob): 
                prob_doc[i] = np.maximum(prob_doc[i], p)
            prob = prob_doc
        elif agg == "mean":
            prob_doc = np.zeros((len(uniq), n_labels))
            counts = np.zeros(len(uniq))
            for i, p in zip(inv, prob): 
                prob_doc[i] += p 
                counts[i] += 1
            prob = prob_doc / counts[:, None]
        pred_bin = (prob >= thr).astype(int)
        ids = uniq

    labels = [[mlb.classes_[j] for j in row.nonzero()[0]] for row in pred_bin]
    return {"id": ids, "prob": prob, "predicted_bin": pred_bin, "predicted_labels": labels}

def predict_df(df_exploded, clf:OneVsRestClassifier, thr, agg = "max"):
    needed_columns = ["id", "chunks", "chunks_embeddings", "chunk_id"]
    if not set(needed_columns).issubset(df_exploded.columns):
        ValueError(f"Missing columns: {[x for x in needed_columns if x not in df_exploded.columns]}")
    
    df_agg = (
        df_exploded.group_by("id").agg([
                *[
                    pl.col(col).first()
                    for col in df_exploded.columns 
                    if col not in ["id", "chunks", "chunks_embeddings", "chunk_id"]
                ],
                pl.col("chunks"),
                pl.col("chunks_embeddings"),
            ])
        )
    
    id_em = dict(zip(df_agg['id'].to_list(), df_agg['chunks_embeddings'].to_list()))
    logger.info('Start prediction.') 
    id_labels = []
    for i, (id_,emb) in tqdm(enumerate(id_em.items()), total = len(id_em.keys()), desc="Predicting docs"): 
        temp_id_labels = {'id': id_}
        X_temp = np.asarray(emb)
        probs = clf.predict_proba(X_temp)
        if agg == 'max': 
            probs_agg = np.max(probs, axis=0).reshape(1, -1)
        elif agg == 'min': 
            probs_agg = np.min(probs, axis=0).reshape(1, -1)
        elif agg == 'mean':
            probs_agg = np.mean(probs, axis=0).reshape(1, -1)
        elif agg == 'median': 
            probs_agg = np.median(probs, axis=0).reshape(1, -1)
        else: 
            raise NotImplementedError 
        label_i = (probs_agg >= thr).astype(int)
        label_indices = np.where(label_i[0] == 1)[0].tolist()
        temp_id_labels['predicted_labels'] = label_indices
        id_labels.append(temp_id_labels)
    
    id_labels_df = pl.DataFrame(id_labels)
    return df_agg.join(id_labels_df, on = 'id', how = 'left')

if __name__ == "__main__": 
    
    prototype = False
    embedding_models = [
        "text-embedding-3-large", 
        'all-MiniLM-L12-v2', 
        'all-mpnet-base-v2'
        ]

    for embedding_model_name in embedding_models: 
        # Preliminary
        embedding_path = FILE_DIR.parent / f"data/blogs_articles/sv_embeddings_{embedding_model_name}.parquet"
        model_dir = FILE_DIR.parent / "models/cat_boost_models"
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.add(model_dir/'training.log')
        # Data
        train_df, val_df, test_df = data_loading(embedding_path, prototype)
        mlb, (y_train_bin, y_val_bin, y_test_bin) = binarize_y(train_df, val_df, test_df)

        X_train = np.asarray(train_df['chunks_embeddings'].to_list())
        X_val   = np.asarray(val_df['chunks_embeddings'].to_list())
        X_test  = np.asarray(test_df['chunks_embeddings'].to_list())

        # Training
        cat_boost_kwargs = {
            "iterations": 500 if not prototype else 2,
            "learning_rate": 0.1,
            "depth": 6 if not prototype else 1,
            "l2_leaf_reg": 3,
            "loss_function": "Logloss", # binary per label
            "eval_metric": "AUC",
            "random_seed": 42,
            "task_type": "GPU",
            "early_stopping_rounds": 50,
            "logging_level": "Silent", 
            'train_dir': (model_dir / f"catboost_{embedding_model_name}_info")}
        
        
        model_data = train(
            X_train, 
            y_train_bin,
            X_val, 
            y_val_bin, 
            **cat_boost_kwargs)
        
        base, clf, thr = model_data["base"], model_data["clf"], model_data["thr"]

        topic_term_representations = create_term_representations(
            train_df['chunks'].to_list(), 
            y_train_bin, 
            n_topics = len(mlb.classes_), 
            top_k=20, 
            label_names=True)
        model_data['topic_term_representations'] = topic_term_representations
        
        # Prediction on Train, Val and Test
        train_pred = predict(
            train_df['chunks_embeddings'].to_list(), 
            clf, 
            mlb, 
            thr, 
            ids=train_df["id"].to_numpy(), 
            agg="max")
        train_pred_df = pl.DataFrame(train_pred)
        train_pred_df = train_df.join(train_pred_df, on = "id", how = 'left')

        val_pred = predict(
            val_df['chunks_embeddings'].to_list(), 
            clf, 
            mlb, 
            thr, 
            ids=val_df["id"].to_numpy(), 
            agg="max")
        val_pred_df = pl.DataFrame(val_pred)
        val_pred_df = val_df.join(val_pred_df, on = "id", how = 'left')

        test_pred = predict(
            test_df['chunks_embeddings'].to_list(), 
            clf, 
            mlb, 
            thr, 
            ids=test_df["id"].to_numpy(), 
            agg="max")
        test_pred_df = pl.DataFrame(test_pred)
        test_pred_df = test_df.join(test_pred_df, on = "id", how = 'left')

        # Evaluation  
        train_metr = compute_all_metrics(train_pred_df['topic_label'].to_list(), train_pred_df['predicted_labels'].to_list(), get_mapping(id_to_label=True))
        val_metr = compute_all_metrics(val_pred_df['topic_label'].to_list(), val_pred_df['predicted_labels'].to_list(), get_mapping(id_to_label=True))
        test_metr = compute_all_metrics(test_pred_df['topic_label'].to_list(), test_pred_df['predicted_labels'].to_list(), get_mapping(id_to_label=True))
        model_data['train_metr'] = train_metr
        model_data['val_metr'] = val_metr
        model_data['test_metr'] = test_metr
        filtered_scores = {k: v for k, v in test_metr.items() if k != 'per_label_table'}
        logger.info(f"Test Scores: {filtered_scores}")

        plot_score_gt_pred(
            train_pred_df, 
            pl.DataFrame(train_metr["per_label_table"]), 
            pred_col = 'predicted_labels', 
            gt_col='topic_label', 
            save = model_dir / f"cat_boost_{embedding_model_name}_train.png")
        
        plot_score_gt_pred(
            val_pred_df, 
            pl.DataFrame(val_metr["per_label_table"]), 
            pred_col = 'predicted_labels', 
            gt_col='topic_label', 
            save = model_dir / f"cat_boost_{embedding_model_name}_val.png")
        
        plot_score_gt_pred(
            test_pred_df, 
            pl.DataFrame(test_metr["per_label_table"]), 
            pred_col = 'predicted_labels', 
            gt_col='topic_label', 
            save = model_dir / f"cat_boost_{embedding_model_name}_test.png")
        
        # Save a dictionary with classifier, base model, thresholds and topic_term_representations etc.
        joblib.dump(model_data, model_dir / f"cat_boost_{embedding_model_name}.joblib")
        logger.info(f"Saved model and finished runnign pipeline for {embedding_model_name}.")

    logger.info("Finished all!")