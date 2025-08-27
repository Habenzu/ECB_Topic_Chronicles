import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix()) # path to root of project
from loguru import logger
import pickle
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
from src.utils import StopwordCollector
from src.evaluation import topic_quality
from collections import defaultdict

FILE_DIR = Path(__file__).parent

class BERTopicPipeline(): 
    def __init__(self, embedding_model_name:str): 
        self.embedding_model_name = embedding_model_name
        self.data_dir_path = FILE_DIR.parent / 'data'
        self.data = self._load_embedding_data()
        self.data_long = self._long_format(self.data)
        self.doc_ids = self.data_long['id'].to_list()
        self.docs = self.data_long['chunks'].to_list()
        self.embeddings = self._get_embeddings()

    def _long_format(self, 
                     data: pl.DataFrame, 
                     select_cols:list= ['id', 'chunks', 'chunks_embeddings'],
                     explode_columns:list = ['chunks', 'chunks_embeddings']) -> pl.DataFrame:
        """
        Convert the data to long format, which means the explode columns are expanded into multiple rows.
        """
        return data.select(select_cols).explode(explode_columns)
    
    def _load_embedding_data(self) -> pl.DataFrame:
        """
        Load the embedding data for the specified model.
        """
        self.data_path = self.data_dir_path / f'embeddings_{self.embedding_model_name}.parquet'

        if self.data_path.exists(): 
            logger.info(f"Loading embedding data for {self.embedding_model_name} from {self.data_path.as_posix()}")
            data = pl.read_parquet(self.data_path)
            logger.info(f"Loaded {len(data)} rows of embedding data for {self.embedding_model_name}.")
            logger.info(f"Data columns: {data.columns}")
            return data
        else: 
            raise FileNotFoundError(f"Embedding data for {self.embedding_model_name} not found in {self.data_path}. Please create the embeddings first using embdding.py script.")

    def _get_embeddings(self): 
        return np.array([np.array(sub) for sub in self.data_long['chunks_embeddings'].to_list()], dtype=np.float64)

    def load_existing_model(self) -> BERTopic:
        """
        Load the existing BERTopic model for the specified embedding model.
        """
        model_path = FILE_DIR.parent / 'models' / f'{self.embedding_model_name}_topic_model.pkl'
        if model_path.exists():
            logger.info(f"Loading existing BERTopic model from {model_path.as_posix()}")
            self.topic_model = BERTopic.load(model_path)
            self.topics, self.probs = self.topic_model.transform(self.docs, self.embeddings)
        else:
            raise FileNotFoundError(f"BERTopic model for {self.embedding_model_name} not found in {model_path}. Please create the model first using topic_modeling.py script.")

    def get_stopwords(self):
        if not hasattr(self, 'stopwords'):
            self.stopwords = StopwordCollector(
                texts=self.data['content'].to_list(),
                authors=self.data['author'].to_list()).get_stopwords()
        else: 
            return self.stopwords

    def create_bert_model(self, 
                          stopwords=None, 
                          umap_model=None, 
                          hdbscan_model=None, 
                          vectorizer_model=None, 
                          ctfidf_model=None):
        """
        Create a BERTopic model with the specified parameters.
        If no parameters are provided, default models will be used.
        """
        if stopwords is None: 
            stopwords = self.get_stopwords()

        if umap_model is None: 
            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

        if hdbscan_model is None: 
            hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

        if vectorizer_model is None: 
            vectorizer_model = CountVectorizer(
                stop_words=stopwords,
                ngram_range=(1, 3),
                min_df=3,
                max_df=0.8)

        if ctfidf_model is None: 
            ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        logger.info(f'Fitting BERTopic model for {self.embedding_model_name}...')
        self.topic_model = BERTopic(
            umap_model=umap_model, 
            hdbscan_model=hdbscan_model, 
            vectorizer_model=vectorizer_model, 
            ctfidf_model=ctfidf_model)

        self.docs = self.data_long['chunks'].to_list()
        self.embeddings = self._get_embeddings()
        self.topics, self.probs = self.topic_model.fit_transform(self.docs, self.embeddings)
        _ = self.assign_topic_per_doc()
        _ = self.evaluate_topic_quality()

    def evaluate_topic_quality(self, docs:list=None, embeddings:list=None, top_n_words = 10): 
        """
        Get the topic quality metrics for the current BERTopic model.
        """
        if not hasattr(self, 'topic_model'):
            raise ValueError("BERTopic model has not been created yet. Please call create_bert_model()/load_existing_model() first.")
        if docs is None and not hasattr(self, 'docs'):
            self.docs = self.data_long['chunks'].to_list()
            docs = self.docs
        elif hasattr(self, 'docs'):
            docs = self.docs
        else: 
            docs = docs
        if embeddings is None and not hasattr(self, 'embeddings'): 
            self.embeddings = self._get_embeddings()
            embeddings = self.embeddings
        elif hasattr(self, 'embeddings'):
            embeddings = self.embeddings
        else: 
            embeddings = embeddings

        quality = topic_quality(
            model = self.topic_model, 
            documents = docs,
            embeddings = embeddings,
            top_n_words=top_n_words)
        self.topic_quality_per_cluster = quality.pop('per_cluster')
        self.topic_quality_all = quality 
        logger.info('Evaulated the topic quality and assigned to self.topic_qualtiy.')
        return [self.topic_quality_all, self.topic_quality_per_cluster]

    def assign_topic_per_doc(self, doc_ids: list = None, topics: list = None, probs: list = None): 
        """
        Assign one dominant topic per document id based on chunk-level topic probabilities.
        Topic -1 is automatically ignored if its probability is 0.

        Parameters:
        - doc_ids: list of document ids (optional, defaults to self.doc_ids)
        - topics: list of topics corresponding to the document ids (optional, defaults to self.topics)
        - probs: list of probabilities corresponding to the document ids (optional, defaults to self.probs)

        Returns:
        - res: polars.DataFrame with columns ["id", "dominant_topic"]
        """
        
        if doc_ids is None or topics is None or probs is None:
            doc_ids = self.doc_ids
            topics = self.topics
            probs = self.probs
        assert len(doc_ids) == len(topics) == len(probs), "length mismatch"
        
        data = np.array(list(zip(doc_ids, topics, probs)))
        topic_scores = defaultdict(lambda: defaultdict(float))  # id -> topic -> summed score

        for doc_id, topic, prob in data:
            doc_id = int(doc_id)
            topic = int(topic)
            topic_scores[doc_id][topic] += prob

        # Select the topic with max score for each doc_id
        dominant_topics = {
            doc_id: max(topic_dict.items(), key=lambda x: x[1])[0]
            for doc_id, topic_dict in topic_scores.items()
        }

        for doc_id, topic in list(dominant_topics.items()):
            if topic == -1 and len(topic_scores[doc_id]) > 1:
                dominant_topics[doc_id] = max(
                    ((t, s) for t, s in topic_scores[doc_id].items() if t != -1),
                    key=lambda x: x[1]
                )[0]

        res = {int(doc_id): int(topic) for doc_id, topic in dominant_topics.items()}

        res = pl.DataFrame({
            "id": list(dominant_topics.keys()),
            "dominant_topic": list(dominant_topics.values())
        }).cast({"id": pl.Int64, "dominant_topic": pl.Int32})

        if 'dominant_topic' not in  self.data.columns:
            self.data = self.data.join(res, left_on="id",right_on = 'id', how="left")
        else: 
            logger.warning("dominant_topic column already exists in data, overwriting it with new values.")
            self.data = self.data.drop("dominant_topic")
            self.data = self.data.join(res, left_on="id",right_on = 'id', how="left")
        logger.info('Assigned 1 topic per document.')
        return res
    
    def save(self, model_dir: str):
        path = Path(model_dir)
        bert_path = path / 'bertopic.pkl'
        pickle_path = path / 'pipeline.pkl'
        
        if hasattr(self, "topic_model"):
            self.topic_model.save(bert_path)
            temp = self.topic_model
            self.topic_model = None
        else:
            temp = None
        
        with open(pickle_path, "wb") as f:
            pickle.dump(self, f)

        if temp:
            self.topic_model = temp
        logger.info(f"Pipeline saved to {pickle_path} and {bert_path}")

    @classmethod
    def load(cls, model_dir: str):
        path = Path(model_dir)
        bert_path = path / 'bertopic.pkl'
        pickle_path = path / 'pipeline.pkl'

        with open(pickle_path, "rb") as f:
            obj = pickle.load(f)

        if bert_path.exists():
            obj.topic_model = BERTopic.load(bert_path)
        else:
            logger.warning(f"No BERTopic model found at {bert_path}, loading pipeline without topic_model.")

        return obj   
    
if __name__ == "__main__":
    from datetime import datetime

    embedding_models = ['text-embedding-3-large', 'all-mpnet-base-v2', 'all-MiniLM-L12-v2']

    for model_name in embedding_models:
        startime = datetime.now()
        model_dir = Path(f'models/topic_model_{model_name}')
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting processing the pipeline for the embedding model {model_name}...")
        topic_pipeline = BERTopicPipeline(
            embedding_model_name=model_name)
        topic_pipeline.create_bert_model()
        topic_pipeline.save(model_dir= model_dir)
        logger.info(f"Finished model training for embedding model {model_name}:")
        logger.info(f"Training time: {datetime.now()-startime}")
        logger.info(f"Model Directory: {model_dir.as_posix()}")
        try: 
            pipeline_load = BERTopicPipeline.load(model_dir)
            logger.success("Sucessfully trained and loaded the model.")
        except Exception as e: 
            logger.error(f"Failed loading the pipeline: {e}")