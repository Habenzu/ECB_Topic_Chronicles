import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from pathlib import Path
import polars as pl
from tqdm import tqdm
from loguru import logger
from src.bertopic_pipeline import BERTopicPipeline
from src.utils import get_mapping
from src.evaluation import compute_all_metrics, plot_topic_metric_per_model, plot_metrics_per_model

FILE_DIR = Path(__file__).parent

class TopicMapper: 
    def __init__(self, train_pred_topics:list, train_gt_topics:list, val_pred_topics:list, val_gt_topics:list,): 
        self.train_pred_topics = train_pred_topics
        self.train_gt_topics = train_gt_topics
        self.val_pred_topics = val_pred_topics
        self.val_gt_topics = val_gt_topics
    
        all_topic_ids = [-1] + list(range(101))
        self.topic_index = {tid: j for j, tid in enumerate(sorted(all_topic_ids))}
        # Create hist vectors of predicted topics & binarize the ground truth: 
        self.T_train = self.to_hist_vector(self.train_pred_topics, self.topic_index)    # (N_tr, K)
        self.y_train = self.binarize_topics(self.train_gt_topics, n_topics=102)         # (N_tr, L)
        self.T_val = self.to_hist_vector(self.val_pred_topics, self.topic_index)        # (N_val, K)
        self.y_val = self.binarize_topics(self.val_gt_topics, n_topics=102)             # (N_val, L)

    def to_hist_vector(self, topic_ids, topic_index):
        K = len(topic_index)
        def f(ts):
            idx = [topic_index[t] for t in ts if t in topic_index]
            v = np.bincount(idx, minlength=K).astype(float)
            s = v.sum()
            return v/s if s else v
        if topic_ids and isinstance(topic_ids[0], (list, tuple, np.ndarray)):
            return np.vstack([f(ts) for ts in topic_ids])
        return f(topic_ids)

    def binarize_topics(self, label_ids, n_topics=102):
        def g(ls):
            idx = [int(l) for l in set(ls) if 0 <= int(l) < n_topics]
            v = np.zeros(n_topics, int)
            if idx:
                v[idx] = 1
            return v
        
        if label_ids and isinstance(label_ids[0], (list, tuple, np.ndarray)):
            return np.vstack([g(ls) for ls in label_ids])
        return g(label_ids)

    def fit_calibrate(self, alpha = 1.0): 
        """
        1. Fit Ridge regression on the training predictions 
        2. Calibrate treshold on the validation predictions
        alpha : ridge regularization strength
        """
        # fit topic label mapping 
        K = self.T_train.shape[1]
        L = self.y_train.shape[1]
        W = np.zeros((K, L))
        for l in range(L):
            y = self.y_train[:, l]
            if y.sum() == 0 or y.sum() == len(y):
                continue  # skip degenerate labels
            model = Ridge(alpha=alpha, fit_intercept=False, positive=True)
            model.fit(self.T_train, y)
            W[:, l] = model.coef_
        self.W_train = W

        # calibrate thresholds
        self.S_val = self.T_val @ self.W_train 
        tau = np.zeros(L)
        for l in range(L):
            y = self.y_val[:, l]
            if y.sum() == 0:
                tau[l] = np.inf  # never predict this label
                continue
            best_f1, best_t = -1, 0.5
            for t in np.linspace(0.01, 0.99, 50):  # grid of thresholds
                yhat = (self.S_val[:, l] >= t).astype(int)
                f1 = f1_score(y, yhat, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            tau[l] = best_t
        self.tau = tau

    def map_predict(self, pred_topics:list): 
        T_pred = self.to_hist_vector(pred_topics, self.topic_index)
        S_pred = T_pred @ self.W_train
        Y_pred = (S_pred >= self.tau).astype(int)
        mapped_topics = [np.flatnonzero(row).tolist() for row in Y_pred]

        return mapped_topics
    
if __name__ == "__main__": 
    DATA_DIR = FILE_DIR.parent / 'data/blogs_articles'
    MODEL_DIR = FILE_DIR.parent / 'models'
    embedding_models = ['text-embedding-3-large', 'all-mpnet-base-v2', 'all-MiniLM-L12-v2']

    def add_preds(df, topic_model):
        preds = []
        for r in tqdm(df.iter_rows(named=True), total =df.shape[0], desc="docs", position=1, leave=False):
            topics, probs = topic_model.transform(documents=r['chunks'], embeddings=np.array(r['chunks_embeddings']))
            preds.append({'id': r['id'], 'pred_topics': topics, 'pred_probs': probs})
        return df.join(pl.DataFrame(preds), on='id', how='left')

    models = {}
    train_set = set(pl.read_parquet(DATA_DIR / "train_topics.parquet")['id'].to_list())
    val_set   = set(pl.read_parquet(DATA_DIR / "val_topics.parquet")['id'].to_list())
    test_set  = set(pl.read_parquet(DATA_DIR / "test_topics.parquet")['id'].to_list())

    for name in tqdm(embedding_models, total=len(embedding_models), desc='models', position=0):
        pipeline = BERTopicPipeline.load(MODEL_DIR / f'sv_topic_model_{name}')

        all_temp = pl.read_parquet(DATA_DIR / f"sv_embeddings_{name}.parquet")
        train_df = add_preds(all_temp.filter(pl.col("id").is_in(train_set)), pipeline.topic_model, )
        val_df   = add_preds(all_temp.filter(pl.col("id").is_in(val_set)), pipeline.topic_model)
        test_df  = add_preds(all_temp.filter(pl.col("id").is_in(test_set)), pipeline.topic_model)

        models[name] = {
            'pipeline': pipeline, 
            'train_df': train_df, 
            'val_df': val_df, 
            'test_df': test_df}
    
    models_ext = models
    for name in tqdm(embedding_models, total = len(embedding_models), desc = 'model', position=0):
        temp_topic_mapper = TopicMapper(
            train_pred_topics = models[name]['train_df']['pred_topics'].to_list(), 
            train_gt_topics = models[name]['train_df']['topic_label'].to_list(),
            val_pred_topics = models[name]['val_df']['pred_topics'].to_list(), 
            val_gt_topics = models[name]['val_df']['topic_label'].to_list())
        temp_topic_mapper.fit_calibrate(alpha = 1.0)
        for part in tqdm(['train', 'val', 'test'], total = 3, desc = 'datasets', position = 1, leave = False):
            temp_mappings = temp_topic_mapper.map_predict(models[name][f'{part}_df']['pred_topics'].to_list())
            train_df = models[name][f'{part}_df'].with_columns(pl.Series("pred_mapped", temp_mappings))
            all_metrics = compute_all_metrics(
                models[name][f'{part}_df']['topic_label'].to_list(),
                temp_mappings, 
                get_mapping(id_to_label=True))
            per_label_table = all_metrics.pop('per_label_table')
            models_ext[name][f"{part}_df"] = train_df
            models_ext[name][f"{part}_all_metrics"] = all_metrics
            models_ext[name][f"{part}_per_label_table"] = pl.DataFrame(per_label_table)

    # PREDICTED TOPICS AND MAPPINGS
    for name in embedding_models: 
        for part in ['train', 'val', 'test']:
            df:pl.DataFrame = models[name][f'{part}_df']
            df = df.with_columns(
                pl.col("pred_probs").map_elements(
                    lambda x: x.tolist() if hasattr(x, "tolist") else x,
                    return_dtype=pl.List(pl.List(pl.Float64))))
            df.write_parquet(MODEL_DIR / f'topic_model_{name}' / f"{part}_df_pred.parquet" )

    # TOPIC METRICS PER MODEL AND DATASET
    for name in embedding_models: 
        for part in ['train', 'val', 'test']:
            models_ext[name][f"{part}_per_label_table"].write_csv(
                MODEL_DIR / f'topic_model_{name}' / f'topic_model_metrics_{part}.csv')
            
    # TOPIC METRICS PER MODEL AND DATASET
    for part in ['train', 'val', 'test']:
        res_dfs = {name: models_ext[name][f"{part}_df"] for name in embedding_models}
        metr_dfs = {name: models_ext[name][f"{part}_per_label_table"] for name in embedding_models}
        for m in ['support', 'precision', 'recall', 'f1', 'balanced_accuracy']:
            plot_topic_metric_per_model(
                res_dfs, 
                metr_dfs, 
                metric = m, 
                top_k = 30, 
                pred_col = "pred_mapped", 
                gt_col = "topic_label",
                save = MODEL_DIR / f"{part}_per_topic_{m}_all_models.png")
            
    # PERFORMANCE METRICS PER MODEL AND DATASET
    metrics_all = {}
    metrics_all['metric'] = list(sorted(models_ext['text-embedding-3-large']['train_all_metrics'].keys()))
    for part in ['train', 'val', 'test']:
        for name in embedding_models: 
            temp_dict = dict(sorted(models_ext[name][f'{part}_all_metrics'].items()))
            metrics_all[name] = list(temp_dict.values())
        metrics_all_df = pl.DataFrame(metrics_all)

        metrics_all_df.write_csv(MODEL_DIR / f'topic_model_all_metrics_per_model_{part}.csv')
        plot_metrics_per_model(metrics_all_df, save = MODEL_DIR / f'{part}_all_metrics_per_model.png')