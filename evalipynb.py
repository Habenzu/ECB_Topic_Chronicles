# %%
from src.bertopic_pipeline import BERTopicPipeline
import polars as pl
from pathlib import Path
from src.utils import plot_cluster_metrics, interactive_plot_model_metrics, plot_topic_trends

embedding_models = [
    'text-embedding-3-large', 
    'all-mpnet-base-v2',
    'all-MiniLM-L12-v2']

model_base_dir = Path('models')
topic_models = {}
overall_benchmarkings = {}
for model in embedding_models: 
    tmp_topic_model = BERTopicPipeline(model).load(model_base_dir / f'topic_model_{model}')
    topic_models[model] = tmp_topic_model
    overall_benchmarkings[model] = tmp_topic_model.topic_quality_all

# %%
interactive_plot_model_metrics(overall_benchmarkings)

# %%
for model, pipeline in topic_models.items(): 
    fig = plot_cluster_metrics(
        pipeline.topic_quality_per_cluster,
        top_n_topics=10,
        model_name=model,
        id_to_name = {i:f'{i}:{pipeline.topic_model.get_topic(i)[0][0]}' for i in pipeline.topic_quality_per_cluster.keys()})
    display(fig)

# %%
datetime_casting = lambda df: df.with_columns([pl.col("date").cast(pl.Datetime("ms")).alias("date")])

for model, pipeline in topic_models.items():
    display(pipeline.topic_model.get_topic_info().head(10))
    fig = plot_topic_trends(datetime_casting(pipeline.data))
    fig.show() 

# %% [markdown]
# # Eval Supervised vs Unsupervised

# %%
from pathlib import Path
from src.supervised_training import load_for_inference, predict
from tqdm import tqdm
from src.evaluation import compute_all_metrics
from src.utils import get_mapping

# %%
def markdown_rowwise_best(df_pl: pl.DataFrame, metric_col: str = "metric", minimize: list = None, precision: int = 3) -> str:
    """
    Create Markdown table where the best model per metric row is highlighted in green+bold,
    the second-best in blue+bold.
    """
    if minimize is None:
        minimize = ["hamming_loss"]

    pdf = df_pl.to_pandas()
    metrics = pdf[metric_col].tolist()
    models = [c for c in pdf.columns if c != metric_col]

    lines = []
    header = "| " + metric_col + " | " + " | ".join(models) + " |"
    sep = "|" + " --- |" * (len(models) + 1)
    lines.append(header)
    lines.append(sep)

    for i, m in enumerate(metrics):
        row_vals = pdf.loc[i, models].astype(float)
        # sort ascending if minimize, else descending
        sorted_vals = row_vals.sort_values(ascending=(m in minimize))
        best_val = sorted_vals.iloc[0]
        second_val = sorted_vals.iloc[1] if len(sorted_vals) > 1 else None

        cells = []
        for c in models:
            val = f"{row_vals[c]:.{precision}f}"
            if row_vals[c] == best_val:
                val = f"<span style='color:green;font-weight:bold'>{val}</span>"
            elif second_val is not None and row_vals[c] == second_val:
                val = f"<span style='color:blue;font-weight:bold'>{val}</span>"
            cells.append(val)
        line = "| " + str(m) + " | " + " | ".join(cells) + " |"
        lines.append(line)

    return "\n".join(lines)

# %%
DATA_DIR = Path('data/blogs_articles')
MODEL_DIR = Path('models')
test_df = pl.read_parquet(DATA_DIR / "test_topics.parquet")
test_df = test_df.sort(by='id')

# %%
# fine tuned roberta eval
ft_path = Path(r'models\experiment_202509050949') / 'supervise_finetune_model_202509050949_baseline'
preds = predict(test_df['text'].to_list(), ft_path, 'cuda', use_chunking=True, agg = 'max')
test_pred = pl.DataFrame(preds).with_columns(pl.Series("id", test_df['id'].to_list()))
test_df_pred = test_df.join(test_pred, on = 'id', how ='left')
ft_all_metrics = compute_all_metrics(test_df_pred['topic_label'].to_list(), test_df_pred['predicted_labels'].to_list(), get_mapping(id_to_label=True))
ft_per_label_table = pl.DataFrame(ft_all_metrics.pop('per_label_table'))
ft_all_metrics_df = pl.DataFrame({
    "metric": list(ft_all_metrics.keys()),
    "finetuned-xlm_roberta_base": list(ft_all_metrics.values())}, 
    strict = False)

# %%
# Only Test
usv_metrics = pl.read_csv(MODEL_DIR / 'topic_model_all_metrics_per_model_test.csv')
usv_metrics = usv_metrics.rename({x:f'usv-{x}' for x in usv_metrics.columns if x != 'metric'})
cb_metrics = pl.read_csv(MODEL_DIR / 'cat_boost_models/all_metrics_per_model.csv')
cb_metrics = cb_metrics.rename({x:f'cb-{x}' for x in cb_metrics.columns if x != 'metric'})


all_metrics = usv_metrics.join(cb_metrics, on='metric', how = 'left').join(ft_all_metrics_df, on = 'metric', how = 'left')
all_metrics.write_csv(MODEL_DIR / 'all_metrics_all_models.csv')

# %%
md_table = markdown_rowwise_best(all_metrics, metric_col="metric", minimize=["hamming_loss"], precision=3)
print(md_table)

# %% [markdown]
# | metric | usv-text-embedding-3-large | usv-all-mpnet-base-v2 | usv-all-MiniLM-L12-v2 | cb-text-embedding-3-large | cb-all-MiniLM-L12-v2 | cb-all-mpnet-base-v2 | finetuned-xlm_roberta_base |
# | --- | --- | --- | --- | --- | --- | --- | --- |
# | example_f1 | 0.491 | 0.523 | 0.473 | <span style='color:green;font-weight:bold'>0.726</span> | 0.486 | 0.643 | <span style='color:blue;font-weight:bold'>0.654</span> |
# | example_precision | 0.449 | 0.503 | 0.393 | <span style='color:green;font-weight:bold'>0.748</span> | 0.381 | 0.575 | <span style='color:blue;font-weight:bold'>0.670</span> |
# | example_recall | 0.674 | 0.684 | 0.759 | 0.765 | <span style='color:green;font-weight:bold'>0.947</span> | <span style='color:blue;font-weight:bold'>0.881</span> | 0.711 |
# | hamming_loss | 0.054 | 0.045 | 0.060 | <span style='color:green;font-weight:bold'>0.019</span> | 0.099 | 0.046 | <span style='color:blue;font-weight:bold'>0.029</span> |
# | jaccard | 0.397 | 0.430 | 0.364 | <span style='color:green;font-weight:bold'>0.653</span> | 0.375 | 0.548 | <span style='color:blue;font-weight:bold'>0.581</span> |
# | macro_balanced_accuracy | 0.622 | 0.673 | 0.672 | 0.648 | <span style='color:green;font-weight:bold'>0.768</span> | <span style='color:blue;font-weight:bold'>0.734</span> | 0.692 |
# | macro_f1 | 0.226 | 0.254 | 0.251 | <span style='color:blue;font-weight:bold'>0.329</span> | 0.247 | 0.309 | <span style='color:green;font-weight:bold'>0.330</span> |
# | macro_precision | 0.241 | 0.246 | 0.221 | <span style='color:green;font-weight:bold'>0.397</span> | 0.177 | 0.244 | <span style='color:blue;font-weight:bold'>0.357</span> |
# | macro_recall | 0.289 | 0.382 | 0.400 | 0.306 | <span style='color:green;font-weight:bold'>0.644</span> | <span style='color:blue;font-weight:bold'>0.516</span> | 0.404 |
# | micro_f1 | 0.403 | 0.466 | 0.425 | <span style='color:green;font-weight:bold'>0.699</span> | 0.383 | 0.551 | <span style='color:blue;font-weight:bold'>0.601</span> |
# | micro_precision | 0.317 | 0.382 | 0.310 | <span style='color:green;font-weight:bold'>0.722</span> | 0.241 | 0.404 | <span style='color:blue;font-weight:bold'>0.548</span> |
# | micro_recall | 0.553 | 0.598 | 0.679 | 0.677 | <span style='color:green;font-weight:bold'>0.939</span> | <span style='color:blue;font-weight:bold'>0.864</span> | 0.665 |
# | n_labels | 102 | 102 | 102 | 102 | 102 | 102 | 102 |
# | n_samples | 525 | 525 | 525 | 525 | 525 | 525 | 525 |
# | subset_accuracy | 0.177 | 0.200 | 0.116 | <span style='color:green;font-weight:bold'>0.419</span> | 0.130 | 0.307 | <span style='color:blue;font-weight:bold'>0.370</span> |
# | weighted_f1 | 0.480 | 0.536 | 0.566 | <span style='color:green;font-weight:bold'>0.668</span> | 0.491 | 0.599 | <span style='color:blue;font-weight:bold'>0.632</span> |
# | weighted_precision | 0.487 | 0.540 | 0.520 | <span style='color:green;font-weight:bold'>0.690</span> | 0.353 | 0.478 | <span style='color:blue;font-weight:bold'>0.641</span> |
# | weighted_recall | 0.553 | 0.598 | 0.679 | 0.677 | <span style='color:green;font-weight:bold'>0.939</span> | <span style='color:blue;font-weight:bold'>0.864</span> | 0.665 |


