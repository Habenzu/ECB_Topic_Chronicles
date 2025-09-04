import os, json
from typing import List, Tuple, Optional

import numpy as np
import polars as pl
from sklearn.metrics import f1_score, precision_recall_fscore_support, precision_score
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from pathlib import Path

FILE_DIR = Path(__file__).parent

# ----------------------------
# Data utilities
# ----------------------------
class MultiLabelDataset(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        tokenizer,
        n_labels: int,
        max_len: int = 512,
        chunk_size: Optional[int] = None,
        chunk_stride: Optional[int] = None
    ):
        self.texts = df["text"].to_list()
        self.labels_list = df["topic_label"].to_list()
        self.n_labels = n_labels
        self.tok = tokenizer
        self.max_len = max_len

        # chunking: precompute (doc_idx, chunk_idx) index if chunking enabled
        self.use_chunking = bool(chunk_size) and chunk_size > 0
        self.chunk_size = chunk_size if self.use_chunking else None
        self.chunk_stride = max(0, chunk_stride) if (self.use_chunking and chunk_stride is not None) else 0
        self.n_chunks = 0
        if self.use_chunking:
            self.doc_chunks = []   # list of dicts returned by tokenizer per doc
            self.index = []        # list of (doc_idx, chunk_idx)
            for di, t in enumerate(self.texts):
                enc = self.tok(
                    t,
                    truncation=True,
                    max_length=self.chunk_size,
                    stride=self.chunk_stride,
                    return_overflowing_tokens=True,
                    padding=False,
                )
                n = len(enc["input_ids"])
                self.n_chunks = self.n_chunks + n
                self.doc_chunks.append(enc)
                for ci in range(n):
                    self.index.append((di, ci))

        logger.info(f"Initialized MultiLabelDataset (chunking={self.use_chunking})\nn_docs {len(self.texts)}\nn_chunks: {self.n_chunks}")

    def __len__(self):
        # number of chunks vs. number of docs
        return len(self.index) if self.use_chunking else len(self.texts)

    def __getitem__(self, i):
        if self.use_chunking:
            di, ci = self.index[i]
            encs = self.doc_chunks[di]
            input_ids = encs["input_ids"][ci]
            attention_mask = encs["attention_mask"][ci]
            y = torch.zeros(self.n_labels, dtype=torch.float)
            for lab in self.labels_list[di]:
                if 0 <= lab < self.n_labels:
                    y[lab] = 1.0
            # include doc_index for later aggregation
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": y, "doc_index": di}
        else:
            enc = self.tok(self.texts[i], truncation=True, padding=False, max_length=self.max_len)
            y = torch.zeros(self.n_labels, dtype=torch.float)
            for lab in self.labels_list[i]:
                if 0 <= lab < self.n_labels:
                    y[lab] = 1.0
            return {**enc, "labels": y}


def collate_pad(batch, pad_id: int):
    keys = ["input_ids", "attention_mask"]
    max_len = max(len(b["input_ids"]) for b in batch)
    for b in batch:
        for k in keys:
            pad_len = max_len - len(b[k])
            if pad_len > 0:
                b[k] = b[k] + ([pad_id] * pad_len)
    input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    attn = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    labels = torch.stack([b["labels"] for b in batch])
    out = {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

    # chunking: pass through doc_index if present
    if "doc_index" in batch[0]:
        out["doc_index"] = torch.tensor([b["doc_index"] for b in batch], dtype=torch.long)
    return out


# ----------------------------
# Imbalance helpers (unchanged)
# ----------------------------
def compute_label_stats(labels_list: List[List[int]], n_labels: int) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(n_labels, dtype=np.int64)
    for labs in labels_list:
        for l in labs:
            if 0 <= l < n_labels:
                counts[l] += 1
    N = len(labels_list)
    prevalence = np.clip(counts / max(N, 1), 1e-8, 1 - 1e-8)
    return counts, prevalence

def make_pos_weights(prevalence: np.ndarray, alpha: float = 0.75, max_w: float = 30.0) -> torch.Tensor:
    w = ((1.0 - prevalence) / prevalence) ** alpha
    w = np.clip(w, 1.0, max_w)
    return torch.tensor(w, dtype=torch.float)

def make_doc_weights(labels_list: List[List[int]], label_weights: np.ndarray, pow_m: float = 0.5) -> np.ndarray:
    n = len(labels_list)
    doc_w = np.zeros(n, dtype=np.float32)
    for i, labs in enumerate(labels_list):
        if len(labs) == 0:
            doc_w[i] = 1.0
        else:
            lw = [label_weights[l] for l in labs if 0 <= l < len(label_weights)]
            doc_w[i] = max(lw) if lw else 1.0
    doc_w = np.power(doc_w, pow_m)
    doc_w = doc_w / max(doc_w.mean(), 1e-6)
    return doc_w


# ----------------------------
# Model (unchanged)
# ----------------------------
class MultiLabelHead(nn.Module):
    def __init__(self, backbone_name: str, n_labels: int, dropout: float = 0.2, use_mean_pool: bool = False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden, n_labels)
        self.use_mean_pool = use_mean_pool

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if self.use_mean_pool:
            mask = attention_mask.unsqueeze(-1).float()
            x = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        else:
            x = token_emb[:, 0]  # CLS
        logits = self.classifier(self.dropout(x))
        return logits


# ----------------------------
# Optional loss & threshold tuning (unchanged)
# ----------------------------
class ASLBCE(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, reduction="mean"):
        super().__init__()
        self.gp = gamma_pos
        self.gn = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits, targets):
        x = torch.sigmoid(logits)
        if self.clip and self.clip > 0:
            x = torch.clamp(x - self.clip, min=0.0, max=1.0)
        xs_pos = x
        xs_neg = 1.0 - x
        lossp = - targets * torch.pow(1 - xs_pos, self.gp) * torch.log(xs_pos.clamp_min(1e-8))
        lossn = - (1 - targets) * torch.pow(1 - xs_neg, self.gn) * torch.log(xs_neg.clamp_min(1e-8))
        loss = lossp + lossn
        return loss.mean() if self.reduction == "mean" else (loss.sum() if self.reduction == "sum" else loss)

def tune_thresholds_with_precision(
    probs: np.ndarray,
    y_true: np.ndarray,
    grid: Optional[np.ndarray] = None,
    precision_floor: Optional[np.ndarray] = None
) -> np.ndarray:
    N, L = probs.shape
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    th = np.zeros(L, dtype=np.float32)
    for l in range(L):
        y = y_true[:, l]; p = probs[:, l]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (p >= t).astype(int)
            if precision_floor is not None:
                prec = precision_score(y, pred, zero_division=0)
                if prec < precision_floor[l]:
                    continue
            f1 = 1.0 if (y.sum()==0 and pred.sum()==0) else f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        th[l] = best_t
    return th


# ----------------------------
# Aggregation helper
# ----------------------------
def _aggregate_doc_probs(per_doc_probs: List[List[np.ndarray]],
                         per_doc_true: List[List[np.ndarray]],
                         agg: str = "max") -> Tuple[np.ndarray, np.ndarray]:
    agg_fn = np.max if agg == "max" else np.mean
    probs_doc = np.stack([agg_fn(np.stack(pp, axis=0), axis=0) if len(pp)>0 else np.zeros_like(per_doc_true[0][0])
                          for pp in per_doc_probs])
    true_doc  = np.stack([tt[0] if len(tt)>0 else np.zeros_like(per_doc_true[0][0]) for tt in per_doc_true])
    return probs_doc, true_doc


# ----------------------------
# Eval helper
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device, logit_adjust=None, *, aggregate: Optional[str] = None, n_docs: Optional[int] = None):
    """
    If aggregate is None: returns stacked chunk-level (or doc-level) probs/true like before.
    If aggregate in {"max","mean"} and n_docs is provided: aggregates chunk predictions per original document.
    """
    model.eval()

    # Chunked document-level path
    if aggregate is not None:
        assert n_docs is not None, "n_docs required when using aggregation"
        per_doc_probs: List[List[np.ndarray]] = [[] for _ in range(n_docs)]
        per_doc_true:  List[List[np.ndarray]] = [[] for _ in range(n_docs)]

        for batch in loader:
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            y   = batch["labels"].cpu().numpy()
            logits = model(ids, att)
            if logit_adjust is not None:
                logits = logits - logit_adjust
            probs = torch.sigmoid(logits).cpu().numpy()

            doc_idx = batch.get("doc_index", None)
            if doc_idx is None:
                raise ValueError("Aggregation requested but no doc_index found in batch.")
            doc_idx = doc_idx.cpu().numpy().tolist()

            for p, t, di in zip(probs, y, doc_idx):
                per_doc_probs[di].append(p)
                per_doc_true[di].append(t)

        return _aggregate_doc_probs(per_doc_probs, per_doc_true, agg=aggregate)

    # Original (non-aggregated) path
    all_probs, all_true = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        y   = batch["labels"].cpu().numpy()
        logits = model(ids, att)
        if logit_adjust is not None:
            logits = logits - logit_adjust
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_true.append(y)
    return np.vstack(all_probs), np.vstack(all_true)


# ----------------------------
# Training (plain method call; prevalence/weighting selectable)
# ----------------------------
def train(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    out_dir: str,
    backbone: str = "xlm-roberta-base",
    n_labels: int = 110,
    max_len: int = 512,
    batch_size: int = 16,
    epochs: int = 4,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_frac: float = 0.06,
    dropout: float = 0.2,
    mean_pool: bool = False,
    device: Optional[str] = None,
    # ---- Imbalance / prevalence toggles ----
    use_prevalence: bool = True,
    use_pos_weight: bool = True,
    use_weighted_sampler: bool = True,
    pos_alpha: float = 0.75,
    pos_maxw: float = 30.0,
    sampler_pow: float = 0.5,
    # ---- Logit adjustment & thresholds ----
    use_logit_adjust: bool = False,
    adjust_in_eval_only: bool = True,
    precision_floor_base: float = 0.0,
    # ---- Loss choice ----
    use_asl: bool = False, asl_gpos: float = 0.0, asl_gneg: float = 4.0, asl_clip: float = 0.05,
    # ---- Chunking toggles ----
    use_chunking: bool = False,
    chunk_size: int = 512,
    chunk_stride: int = 64,
    agg: str = "max",
    init_from_best: Optional[str] = None,):
    """
    Returns: dict with keys: best_ckpt_path, thresholds_path, prevalence_path (optional), metrics
    """
    logger.info("Starting training run")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(out_dir
    ).mkdir(parents=True, exist_ok=True)

    assert {"text", "topic_label"}.issubset(train_df.columns), "DataFrames must have columns: text, topic_label"

    # ---- Prevalence ----
    need_prev = use_prevalence and (use_pos_weight or use_weighted_sampler or use_logit_adjust or precision_floor_base > 0.0)
    train_labels = train_df["topic_label"].to_list()
    prevalence = None
    if need_prev:
        _, prevalence = compute_label_stats(train_labels, n_labels)

    # ---- Tokenizer & datasets ----
    tok = AutoTokenizer.from_pretrained(backbone)

    # Use chunked datasets if requested
    train_ds = MultiLabelDataset(
        train_df, tok, n_labels=n_labels,
        max_len=max_len,
        chunk_size=(chunk_size if use_chunking else None),
        chunk_stride=(chunk_stride if use_chunking else None),
    )
    val_ds = MultiLabelDataset(
        val_df, tok, n_labels=n_labels,
        max_len=max_len,
        chunk_size=(chunk_size if use_chunking else None),
        chunk_stride=(chunk_stride if use_chunking else None),
    )
    test_ds = MultiLabelDataset(
        test_df, tok, n_labels=n_labels,
        max_len=max_len,
        chunk_size=(chunk_size if use_chunking else None),
        chunk_stride=(chunk_stride if use_chunking else None),
    )

    # ---- Sampler / loaders ----
    if use_weighted_sampler and prevalence is not None:
        label_w = (1.0 - prevalence) / prevalence
        doc_w = make_doc_weights(train_labels, label_w, pow_m=sampler_pow)

        # chunking: sample weights at the *sample* level if chunking
        if getattr(train_ds, "use_chunking", False):
            sample_w = [float(doc_w[di]) for (di, _ci) in train_ds.index]
            weights = torch.tensor(sample_w, dtype=torch.double)
            num_samples = len(train_ds)
        else:
            weights = torch.tensor(doc_w, dtype=torch.double)
            num_samples = len(train_ds)

        sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
        shuffle_train = False
    else:
        sampler = None
        shuffle_train = True

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle_train,
        collate_fn=lambda b: collate_pad(b, tok.pad_token_id)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_pad(b, tok.pad_token_id)
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_pad(b, tok.pad_token_id)
    )

    # ---- Model, loss, optim, sched ----
    model = MultiLabelHead(backbone, n_labels=n_labels, dropout=dropout, use_mean_pool=mean_pool).to(device)
    # Warm-start from previous best checkpoint (optional)
    if init_from_best is not None:
        prev_dir = Path(init_from_best)
        ckpt_path = prev_dir / "best_model.pt"
        if ckpt_path.exists():
            # (Optional) sanity check backbone match if file present
            bb_json = prev_dir / "backbone.json"
            if bb_json.exists():
                prev_bb = json.load(open(bb_json, "r")).get("backbone", backbone)
                if prev_bb != backbone:
                    logger.warning(f"[init_from_best] Backbone mismatch: prev='{prev_bb}' vs now='{backbone}'. Proceeding anyway.")
            logger.info(f"[init_from_best] Loading weights from {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            logger.warning(f"[init_from_best] No checkpoint found at {ckpt_path}; training from scratch.")

    if use_asl:
        criterion = ASLBCE(gamma_pos=asl_gpos, gamma_neg=asl_gneg, clip=asl_clip)
        pos_w_t = None
    else:
        if use_pos_weight and prevalence is not None:
            pos_w_t = make_pos_weights(prevalence, alpha=pos_alpha, max_w=pos_maxw).to(device)
        else:
            pos_w_t = None
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * max(epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_frac * total_steps),
        num_training_steps=total_steps
    )

    # ---- Optional logit adjustment vector ----
    logit_adjust_vec = None
    if use_logit_adjust and prevalence is not None:
        la = np.log(prevalence / (1.0 - prevalence))
        logit_adjust_vec = torch.tensor(la, dtype=torch.float, device=device)

    if init_from_best is not None: 
        best_ckpt = os.path.join(out_dir, "best_model.pt")
        # Start with current model’s validation F1 as the baseline to beat
        with torch.no_grad():
            start_val_probs, start_val_true = evaluate(
                model, val_loader, device,
                logit_adjust=(None if not (use_logit_adjust and adjust_in_eval_only) else torch.tensor(
                    np.log(prevalence/(1.0 - prevalence)), dtype=torch.float, device=device
                ) if (use_logit_adjust and (prevalence is not None)) else None),
                aggregate=(agg if use_chunking else None),
                n_docs=(len(val_df) if use_chunking else None)
            )
            start_floors = None
            if precision_floor_base > 0.0:
                if prevalence is not None:
                    ranks = prevalence.argsort().argsort()
                    rarity = 1.0 - (ranks / (n_labels - 1 + 1e-8))
                    start_floors = (precision_floor_base * rarity).clip(0.0, 0.9)
                else:
                    start_floors = np.full(n_labels, precision_floor_base, dtype=np.float32)
            start_th = tune_thresholds_with_precision(start_val_probs, start_val_true, precision_floor=start_floors)
            start_pred = (start_val_probs >= start_th).astype(int)
            best_micro = f1_score(start_val_true, start_pred, average="micro")
        logger.info(f"Starting from micro-F1={best_micro:.4f} (pre-training validation)")
    else: 
        best_micro = -1.0
        best_ckpt = os.path.join(out_dir, "best_model.pt")

    # ---- Train loop ----
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {ep}", leave=False):
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            y   = batch["labels"].to(device)

            logits = model(ids, att)
            if logit_adjust_vec is not None and not adjust_in_eval_only:
                logits = logits - logit_adjust_vec

            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()

        # ---- Validation ----
        val_probs, val_true = evaluate(
            model, val_loader, device,
            logit_adjust=(logit_adjust_vec if adjust_in_eval_only else None),
            aggregate=(agg if use_chunking else None),
            n_docs=(len(val_df) if use_chunking else None)
        )

        # Precision floors
        if precision_floor_base > 0.0:
            if prevalence is not None:
                ranks = prevalence.argsort().argsort()
                rarity = 1.0 - (ranks / (n_labels - 1 + 1e-8))
                floors = (precision_floor_base * rarity).clip(0.0, 0.9)
            else:
                floors = np.full(n_labels, precision_floor_base, dtype=np.float32)
        else:
            floors = None

        th = tune_thresholds_with_precision(val_probs, val_true, precision_floor=floors)
        val_pred = (val_probs >= th).astype(int)
        micro = f1_score(val_true, val_pred, average="micro")
        macro = f1_score(val_true, val_pred, average="macro")
        print(f"Epoch {ep}/{epochs} - loss {running/len(train_loader):.4f} | micro-F1 {micro:.4f} | macro-F1 {macro:.4f}")

        if micro > best_micro:
            best_micro = micro
            torch.save(model.state_dict(), best_ckpt)
            np.save(os.path.join(out_dir, "thresholds.npy"), th)
            with open(os.path.join(out_dir, "backbone.json"), "w") as f:
                json.dump({"backbone": backbone}, f)
            if prevalence is not None:
                np.save(os.path.join(out_dir, "prevalence.npy"), prevalence)

    # ---- Test ----
    logger.info("Loading best checkpoint for test...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    th = np.load(os.path.join(out_dir, "thresholds.npy"))
    test_probs, test_true = evaluate(
        model, test_loader, device,
        logit_adjust=(logit_adjust_vec if adjust_in_eval_only else None),
        aggregate=(agg if use_chunking else None),          
        n_docs=(len(test_df) if use_chunking else None)     
    )
    test_pred = (test_probs >= th).astype(int)
    micro = f1_score(test_true, test_pred, average="micro")
    macro = f1_score(test_true, test_pred, average="macro")
    logger.info(f"TEST micro-F1 {micro:.4f} | macro-F1 {macro:.4f}")

    prec, rec, f1, support = precision_recall_fscore_support(test_true, test_pred, average=None, zero_division=0)
    per_label = {"precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(), "support": support.tolist()}
    with open(os.path.join(out_dir, "per_label_metrics.json"), "w") as f:
        json.dump(per_label, f)

    return {
        "best_ckpt_path": best_ckpt,
        "thresholds_path": os.path.join(out_dir, "thresholds.npy"),
        "prevalence_path": (os.path.join(out_dir, "prevalence.npy") if need_prev else None),
        "metrics": {"test_micro_f1": float(micro), "test_macro_f1": float(macro)}
    }


# ----------------------------
# Inference helpers
# ----------------------------
def load_for_inference(out_dir: str, device: Optional[str] = None):
    cfg = json.load(open(os.path.join(out_dir, "backbone.json")))
    backbone = cfg["backbone"]
    th = np.load(os.path.join(out_dir, "thresholds.npy"))
    prev_path = os.path.join(out_dir, "prevalence.npy")
    prevalence = np.load(prev_path) if os.path.exists(prev_path) else None
    tok = AutoTokenizer.from_pretrained(backbone)
    model = MultiLabelHead(backbone, n_labels=len(th))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), map_location=dev))
    model.to(dev).eval()
    logit_adjust = None
    if prevalence is not None:
        logit_adjust = torch.tensor(np.log(prevalence/(1-prevalence)), dtype=torch.float, device=dev)
    return tok, model, th, logit_adjust, dev

def predict(
    texts,
    out_dir: str,
    device: Optional[str] = None,
    max_len: int = 512,
    label_names: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    apply_logit_adjust: bool = True,
    threshold_override: Optional[np.ndarray] = None,
    use_chunking: bool = False,
    chunk_size: int = 512,
    chunk_stride: int = 64,
    agg: str = "max",
):
    if isinstance(texts, str):
        texts = [texts]

    tok, model, th, logit_adjust, dev = load_for_inference(out_dir, device=device)
    if threshold_override is not None:
        th = np.asarray(threshold_override, dtype=np.float32)

    outputs = []

    if not use_chunking:
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.no_grad():
            logits = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
            if apply_logit_adjust and logit_adjust is not None:
                logits = logits - logit_adjust
            probs = torch.sigmoid(logits).cpu().numpy()

        preds_bin = (probs >= th).astype(int)
        for i in range(len(texts)):
            on_idx = np.where(preds_bin[i] == 1)[0]
            active = sorted([(j, float(probs[i, j])) for j in on_idx], key=lambda x: x[1], reverse=True)
            sorted_idx = [j for j, _ in active]
            item = {
                "text_index": i,
                "predicted_labels": [label_names[j] if label_names else j for j in sorted_idx],
                "predicted_probs": [float(probs[i, j]) for j in sorted_idx],
                "thresholds_used": [float(th[j]) for j in sorted_idx],
            }
            if top_k is not None:
                order = np.argsort(-probs[i])[:top_k]
                item["top_k"] = [
                    {
                        "label": (label_names[j] if label_names else j),
                        "p": float(probs[i, j]),
                        "t": float(th[j]),
                        "above_threshold": bool(probs[i, j] >= th[j]),
                    } for j in order
                ]
            outputs.append(item)
        return outputs

    # ---- Chunked inference ----
    for i, text in enumerate(texts):
        enc = tok(
            text,
            truncation=True,
            max_length=chunk_size,
            stride=chunk_stride,
            return_overflowing_tokens=True,
            padding=False
        )
        n = len(enc["input_ids"])
        if n == 0:
            # empty fallback
            probs_doc = np.zeros_like(th, dtype=np.float32)
        else:
            # batched over chunks
            ids = torch.tensor(enc["input_ids"], dtype=torch.long, device=dev)
            att = torch.tensor(enc["attention_mask"], dtype=torch.long, device=dev)
            with torch.no_grad():
                logits = model(ids, att)
                if apply_logit_adjust and logit_adjust is not None:
                    logits = logits - logit_adjust
                probs_chunks = torch.sigmoid(logits).cpu().numpy()
            probs_doc = probs_chunks.max(axis=0) if agg == "max" else probs_chunks.mean(axis=0)

        preds_bin = (probs_doc >= th).astype(int)
        on_idx = np.where(preds_bin == 1)[0]
        active = sorted([(j, float(probs_doc[j])) for j in on_idx], key=lambda x: x[1], reverse=True)
        sorted_idx = [j for j, _ in active]
        item = {
            "text_index": i,
            "predicted_labels": [label_names[j] if label_names else j for j in sorted_idx],
            "predicted_probs": [float(probs_doc[j]) for j in sorted_idx],
            "thresholds_used": [float(th[j]) for j in sorted_idx],
        }
        if top_k is not None:
            order = np.argsort(-probs_doc)[:top_k]
            item["top_k"] = [
                {
                    "label": (label_names[j] if label_names else j),
                    "p": float(probs_doc[j]),
                    "t": float(th[j]),
                    "above_threshold": bool(probs_doc[j] >= th[j]),
                } for j in order
            ]
        outputs.append(item)

    return outputs

def _write_args_txt(args: dict, name: str, date_str: str) -> None:
    skip = {"train_df", "val_df", "test_df"}
    out_dir: Path = args["out_dir"]
    
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.txt", "w", encoding="utf-8") as f:
        f.write(f"experiment_name: {name}\nrun_ts: {date_str}\n\n")
        for k, v in args.items():
            if k in skip:
                continue
            if isinstance(v, Path):
                v = str(v)
            f.write(f"{k}: {v}\n")

if __name__ == '__main__': 
    from copy import deepcopy
    from datetime import datetime
    import json

    date = datetime.now().strftime("%Y%m%d%H%M")
    MODEL_DIR = FILE_DIR.parent / 'models'
    EXPERIMENT_DIR = MODEL_DIR / f'experiment_{date}'
    DATA_DIR = FILE_DIR.parent / 'data/blogs_articles'
    
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(EXPERIMENT_DIR / "logging.log")
    
    prototype = False
    tr_path = DATA_DIR / 'train_topics.parquet'
    va_path = DATA_DIR / 'val_topics.parquet'
    te_path = DATA_DIR / 'test_topics.parquet'
    train_df = pl.read_parquet(tr_path).sort(by='filename')
    val_df   = pl.read_parquet(va_path).sort(by='filename')
    test_df  = pl.read_parquet(te_path).sort(by='filename')

    if prototype:
        train_df = train_df.sample(20, seed=42)
        val_df   = val_df.sample(10,  seed=42)
        test_df  = test_df.sample(10, seed=42)

    # ---------------- Baseline args (no balancing) ----------------
    base_args = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "out_dir": EXPERIMENT_DIR / f"supervise_finetune_model_{date}_baseline",

        # Encoder & heads
        "backbone": "xlm-roberta-base",
        "n_labels": 110,
        "dropout": 0.2,
        "mean_pool": True,

        # Sequence & chunking (fixed; no aggregation experiments)
        "use_chunking": True,
        "chunk_size": 512,
        "chunk_stride": 64,
        "agg": "max",

        # Batching & schedule
        "max_len": 512,
        "batch_size": 8,
        "epochs": 2,
        "lr": 2e-5,
        "weight_decay": 0.01,
        "warmup_frac": 0.06,
        "device": "cuda",

        # Imbalance controls (no sampler anywhere)
        "use_prevalence": False,        # off in baseline
        "use_pos_weight": False,
        "pos_alpha": 0.75,
        "pos_maxw": 30.0,
        "use_weighted_sampler": False,  # NEVER enable
        "sampler_pow": 0.5,

        # Logit adjust & thresholds
        "use_logit_adjust": False,
        "adjust_in_eval_only": True,
        "precision_floor_base": 0.0,

        # Loss
        "use_asl": False,
        "asl_gpos": 0.0,
        "asl_gneg": 4.0,
        "asl_clip": 0.05,
        'init_from_best': MODEL_DIR / 'experiment_202509040907' / 'supervise_finetune_model_202509040907_baseline'
    }

    # ---------------- Experiments (baseline + 3 others) ----------------
    # experiments = [
    #     ("baseline", {}),

    #     # 1) Balancing approach A: BCE + pos_weight (needs prevalence)
    #     ("pos_weight", {
    #         "use_pos_weight": True,
    #         "use_prevalence": True,
    #         "out_dir": MODEL_DIR / f"supervise_finetune_model_{date}_posw",
    #     }),

    #     # 2) Balancing approach B: ASL (no pos_weight, no prevalence needed)
    #     ("asl", {
    #         "use_asl": True,
    #         "use_pos_weight": False,
    #         "use_prevalence": False,
    #         "out_dir": MODEL_DIR / f"supervise_finetune_model_{date}_asl",
    #     }),

    #     # 3) Balancing approach C: logit adjustment (+ gentle precision floor), no pos_weight
    #     ("logit_adjust", {
    #         "use_logit_adjust": True,
    #         "use_prevalence": True,
    #         "precision_floor_base": 0.10,
    #         "use_pos_weight": False,
    #         "out_dir": MODEL_DIR / f"supervise_finetune_model_{date}_logadj",
    #     }),
    # ]
    # ---------------- Experiments (baseline + stacked optimizations) ----------------
    experiments = [
        ("baseline", {}),

        # # Stacked: BCE + pos_weight  + logit-adjust (eval) + precision floor for threshold tuning
        # ("stacked", {
        #     "use_prevalence": True,       # needed for priors/weights
        #     "use_pos_weight": True,       # imbalance in the loss
        #     "pos_alpha": 0.75,
        #     "pos_maxw": 30.0,

        #     "use_logit_adjust": True,     # prior correction at eval/inference
        #     "adjust_in_eval_only": True,  # don't alter training gradients
        #     "precision_floor_base": 0.10, # stabilize rare labels during threshold tuning

        #     "use_asl": False,             # keep off to avoid double-correcting with pos_weight
        #     "use_weighted_sampler": False,# never enable per your requirement

        #     "out_dir": EXPERIMENT_DIR / f"supervise_finetune_model_{date}_stacked",
        # }),
    ]
    
    all_results = []
    for name, overrides in experiments:
        args = deepcopy(base_args)
        args.update(overrides)
        if "out_dir" not in overrides:
            args["out_dir"] = EXPERIMENT_DIR / f"supervise_finetune_model_{date}_{name}"

        _write_args_txt(args, name=name, date_str=date)
        logger.info(f"=== Running experiment: {name} ===")
        try:
            res = train(**args)
            all_results.append({
                "name": name,
                "out_dir": str(args["out_dir"]),
                **res.get("metrics", {})
            })
        except RuntimeError as e:
            logger.error(f"Experiment '{name}' failed: {e}")
            all_results.append({
                "name": name,
                "out_dir": str(args["out_dir"]),
                "error": str(e)
            })

    # ---------------- Summary ----------------
    all_results_sorted = sorted(
        all_results,
        key=lambda x: x.get("test_micro_f1", float("-inf")),
        reverse=True
    )

    logger.info("\n=== Experiment summary (by test_micro_f1) ===")
    for r in all_results_sorted:
        if "error" in r:
            print(f"{r['name']:>12} | ERROR: {r['error']}")
        else:
            print(f"{r['name']:>12} | micro: {r['test_micro_f1']:.4f} | macro: {r['test_macro_f1']:.4f} | {r['out_dir']}")

    summary_path = EXPERIMENT_DIR / f"summary_{date}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results_sorted, f, indent=2)
    logger.info(f"\nSaved summary to: {summary_path}")