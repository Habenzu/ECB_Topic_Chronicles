import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix()) # path to root of project

import os
import time
import tiktoken
from collections import deque
from typing import List, Iterable
from openai import OpenAI
import polars as pl
from typing import List
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from langchain_text_splitters import TokenTextSplitter

FILE_DIR = Path(__file__).parent

class Embedder(): 
    def __init__(
        self,
        model_name:str = "text-embedding-3-large",
        api_key:str = None,
        api_key_var:str = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_key_var = api_key_var
        self._set_class_attributes_and_methods()  # wires backend + attrs
        self._request_tstamps: deque[float] = deque()
        self._token_tstamps: deque[tuple] = deque()
        logger.info(f"Embedder initialized with model: {self.model_name}, embedding_dim: {self.embedding_dim}")

    def _prune(self):
        cut = time.time() - 60
        while self._request_tstamps and self._request_tstamps[0] < cut:
            self._request_tstamps.popleft()
        while self._token_tstamps and self._token_tstamps[0][0] < cut:
            self._token_tstamps.popleft()

    def _check_rate(self, tokens:int):
        """
        For OpenAI models, enforce RPM/TPM; for local (Sentence-Transformer) models,
        limits are set to inf so this becomes a fast no-op.
        """
        self._prune()
        while (
            len(self._request_tstamps) >= self.requests_per_minute or
            sum(t for _, t in self._token_tstamps) + tokens > self.tokens_per_minute
        ):
            time.sleep(0.25)
            self._prune()
        now = time.time()
        self._request_tstamps.append(now)
        self._token_tstamps.append((now, tokens))

    # ------------------- batching -------------------
    def _batch_openai(self, texts:List[str]) -> Iterable[List[str]]:
        batch, tok = [], 0
        for t in texts:
            tt = len(self._encode(t))
            if tt > self.batch_token_limit or tt > self.max_tokens:
                raise ValueError("single text exceeds batch_token_limit")
            if tok + tt > self.batch_token_limit and batch:
                yield batch
                batch, tok = [], 0
            batch.append(t)
            tok += tt
        if batch:
            yield batch

    def _batch_local(self, texts:List[str]) -> Iterable[List[str]]:
        """Sentence-Transformer models can accept the full list (they batch internally)."""
        yield texts

    def embed(self, texts: str | List[str], progress_bar = False, log = False) -> List[List[float]]:
        if not isinstance(texts, list):
            texts = [texts]
        out = []
        batches = list(self._batch(texts))  # backend-specific batch fn
        if log: logger.info(f"Starting embedding of {len(texts)} texts in {len(batches)} batches")
        for b in tqdm(batches, desc="Embedding batches", disable=(not progress_bar) or (len(texts) <= 1)):
            toks = self.count_tokens(b, progress_bar=False)  # token count for logging / rate
            self._check_rate(toks)
            out.extend(self._embed_impl(b))
        if log: logger.info(f"Completed embedding of {len(texts)} texts, produced {len(out)} embeddings")
        return out

    # unchanged methods below -----------------------------------------------
    def _set_class_attributes_and_methods(self):
        """
        Wire all backend-specific attributes and function pointers.

        Supported:
          * OpenAI "text-embedding-3-large"  (remote API)
          * Sentence-Transformer "all-MiniLM-L6-v2" (local HF model)

        To support a different Sentence-Transformer model, change the string below.
        """
        # ---------------- OpenAI backend ----------------
        if self.model_name == "text-embedding-3-large": 
            if self.api_key is None:
                if self.api_key_var is not None:
                    self.api_key = os.getenv(self.api_key_var)
                else:
                    raise ValueError("api_key or api_key_var must be defined!")
            if not self.api_key:
                raise ValueError("API key could not be retrieved from environment or was empty.")

            self.max_tokens = 8_191
            self.tiktoken_name = "cl100k_base"
            self.tokenizer = tiktoken.get_encoding(self.tiktoken_name)
            self._embed_impl = self._openai_embedding
            self._batch = self._batch_openai
            self.embedding_dim = 3_072
            self.requests_per_minute = 2_500 
            self.tokens_per_minute  = 1_000_000	
            self.price_per_mio = 0.13
            self.batch_token_limit  = 10 * self.max_tokens
            self.token_splitter = TokenTextSplitter.from_tiktoken_encoder(
                encoding_name=self.tiktoken_name,
                chunk_size=self.max_tokens - self.max_tokens//20, # 5% margin to be on the save side
                chunk_overlap=self.max_tokens//10)

        # ---------------- Sentence-Transformer backend ----------------
        elif self.model_name in ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2"]:
            from sentence_transformers import SentenceTransformer  # lazy import
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self._st_model = SentenceTransformer(self.model_name, device=device)
            self._st_tokenizer = getattr(self._st_model, "tokenizer", None)
            self.embedding_dim = self._st_model.get_sentence_embedding_dimension()
            self.tokenizer = self._encode_sentence_transformer
            self._embed_impl = self._sentence_transformer_embedding
            self._batch = self._batch_local
            self.max_tokens = self._st_model.max_seq_length
            self.token_splitter = TokenTextSplitter.from_huggingface_tokenizer(
                self._st_tokenizer,
                chunk_size=self.max_tokens - self.max_tokens//20, # 5% margin to be on the save side
                chunk_overlap=self.max_tokens//10)

            # local models: no OpenAI pricing / rate limits / max token caps
            self.requests_per_minute = float("inf")
            self.tokens_per_minute = float("inf")
            self.price_per_mio = 0.0
            self.batch_token_limit = float("inf")

        else: 
            raise NotImplementedError(f"Model {self.model_name} not supported")

    def _encode_sentence_transformer(self, text:str):
        """
        Proper tokenization for Sentence-Transformer models:
        use the underlying Hugging Face tokenizer if available; fall back to
        simple whitespace tokenization only as a last resort.
        """
        if getattr(self, "_st_tokenizer", None) is not None:
            # add_special_tokens=False so counts align with "content tokens"
            return self._st_tokenizer.encode(text, add_special_tokens=False)
        return text.split()

    def _encode(self, text:str): 
        return self.tokenizer.encode(text) if hasattr(self.tokenizer, "encode") else self.tokenizer(text)

    def count_tokens(self, texts: str | List[str], progress_bar = False):
        if not isinstance(texts, list):
            texts = [texts]
        return sum(len(self._encode(t)) for t in tqdm(texts, desc="Counting tokens", disable=(not progress_bar) or (len(texts) <= 1)))
    
    def _openai_embedding(self, texts: List[str]) -> List[List[float]]: 
        self._client = OpenAI(api_key=self.api_key)
        r = self._client.embeddings.create(model=self.model_name, input=texts)
        return [d.embedding for d in r.data]

    def _sentence_transformer_embedding(self, texts: List[str]) -> List[List[float]]:
        """
        Local embedding via Sentence-Transformer model already loaded in _set_class_attributes_and_methods.
        encode() will internally batch; we return Python lists of floats to mirror OpenAI shape.
        """
        # convert_to_numpy=False returns list[list[float]]
        return self._st_model.encode(texts, convert_to_numpy=False, show_progress_bar=False)

    def _chunk_text(self, text: str) -> List: 
        """
        Chunk text into smaller parts for embedding.
        """
        chunks = self.token_splitter.split_text(text)
        return chunks

    def chunk_and_embed_df(self, df:pl.DataFrame, text_column:str = 'content', chunk_column:str = 'chunks', id_column:str = 'id', accept:bool = False): 
        # 1. Chunking
        if chunk_column not in df.columns or not isinstance(df[chunk_column].dtype, pl.List): 
            logger.info(f'Creating {chunk_column} column because not chunked yet, or overrriding {chunk_column} column because not dtype List.')
            df = df.with_columns(pl.col(text_column).map_elements(self._chunk_text, return_dtype=pl.List(pl.String)).alias(chunk_column))
        else: 
            logger.info('Already properly chunked.')
        # 2. Embedding
        embeddings = []

        # Quick Price check when using OpenAI
        if self.model_name == 'text-embedding-3-large': 
            token_counts = self.count_tokens([item for sublist in df[chunk_column] for item in sublist])
            logger.info(f'Price for {self.model_name} embeddings: {round(token_counts/1_000_000 * self.price_per_mio, 2)} USD') 
            if not accept:
                response = input(f"Price for {self.model_name} embeddings: {round(token_counts/1_000_000 * self.price_per_mio, 2)} USD \n\nContinue? (y/n): ").strip().lower()
                if response != "y":
                    logger.info("Aborted.")
                    return None

        for row in tqdm(df.select([id_column, chunk_column]).iter_rows(named = True), desc="Embedding chunks", total = len(df)):
            embedded_chunks = self.embed(row[chunk_column])
            if not isinstance(embedded_chunks[0], List): # if not list then its torch.Tensor
                embedded_chunks = [x.tolist() for x in embedded_chunks]
            embeddings.append({id_column:row[id_column], 'chunks_embeddings': embedded_chunks})

        schema = {
            id_column: pl.Int64,
            "chunks_embeddings": pl.List(pl.List(pl.Float64))}

        embeddings_df = pl.DataFrame(embeddings, schema = schema)
        df = df.join(embeddings_df[[id_column, 'chunks_embeddings']], on = id_column, how = 'left')
        return df
    
if __name__ == '__main__': 
    path_to_preprocessed_data = Path(r'data\preprocessed_en\preprocessed_data.jsonl')
    preprocessed_data = pl.read_ndjson(path_to_preprocessed_data)
    logger.info(f'Loaded preprocessed data from {path_to_preprocessed_data.as_posix()}')
    
    # OpenAI
    api_key = Path(FILE_DIR.parent / "openai.key").read_text(encoding="utf-8").strip()
    oa_embedder = Embedder(api_key=api_key)
    oa_embeddings = oa_embedder.chunk_and_embed_df(preprocessed_data, accept=True)
    oa_embeddings.write_parquet(FILE_DIR.parent / "data" / f"embeddings_{oa_embedder.model_name}.parquet")
    logger.info(f"OpenAI embeddings saved to {FILE_DIR.parent / 'data' / f'embeddings_{oa_embedder.model_name}.parquet'}")

    # Sentence-Transformer Models
    st_embedder = Embedder(model_name = 'all-MiniLM-L12-v2')
    st_embeddings = st_embedder.chunk_and_embed_df(preprocessed_data)
    st_embeddings.write_parquet(FILE_DIR.parent / "data" / f"embeddings_{st_embedder.model_name}.parquet")
    logger.info(f"Sentence-Transformer embeddings saved to {FILE_DIR.parent / 'data' / f'embeddings_{st_embedder.model_name}.parquet'}")

    st_embedder_2 = Embedder(model_name = 'all-mpnet-base-v2')
    st_embeddings_2 = st_embedder_2.chunk_and_embed_df(preprocessed_data)
    st_embeddings_2.write_parquet(FILE_DIR.parent / "data" / f"embeddings_{st_embedder_2.model_name}.parquet")
    logger.info(f"Sentence-Transformer embeddings saved to {FILE_DIR.parent / 'data' / f'embeddings_{st_embedder_2.model_name}.parquet'}")

    # Word 2 Vec embedding dim: 768