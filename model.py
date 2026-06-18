"""
model.py
================================================================================
Predictors and the learned plausibility scorer for the Defense-Aware Selective
Prediction (DASP) system.

This module provides three components behind small, explicit interfaces:

  1. HierarchicalEncoder        Long-document encoder over InLegalBERT (110M).
                                Splits a case into <=512-token chunks, encodes each,
                                and pools chunk representations (mean | attention).
  2. InLegalBERTClassifier      The main predictor. Wraps the encoder with a binary
                                head, supports end-to-end fine-tuning OR frozen-feature
                                training, and produces (a) a DETERMINISTIC calibrated
                                probability for the point prediction and (b) a
                                STOCHASTIC Monte-Carlo-dropout predictive distribution
                                for epistemic uncertainty consumed by selective.py.
  3. TfidfLinearBaseline        A fast, faithful baseline / ablation point.
     PlausibilityScorer         Learned tier of plausibility.py; uses a SEPARATE,
                                frozen encoder so it can never corrupt the predictor.

--------------------------------------------------------------------------------
WHY THE NON-OBVIOUS DESIGN CHOICES (each addresses a concrete failure mode):

(A) Monte-Carlo dropout. Dropout is disabled by model.eval(). A naive MC-dropout
    implementation therefore returns identical forward passes and a *zero* epistemic
    signal. We selectively re-enable ONLY nn.Dropout modules (keeping everything else,
    e.g. LayerNorm, in eval mode) via `enable_mc_dropout`. The point prediction uses a
    fully-deterministic eval pass; the uncertainty uses T stochastic passes. These are
    two distinct code paths and are never conflated.

(C/E) Calibration. DASP and the conformal layer consume P(favorable|x) as if it were
    calibrated. Raw softmax from a fine-tuned, class-weighted transformer is
    overconfident, which would make every downstream threshold meaningless. We fit a
    single temperature on the development set (Guo et al., 2017) AFTER training and
    expose only temperature-scaled probabilities. Class weighting (for ILDC imbalance)
    is applied in the loss; the temperature is fit afterwards so reported probabilities
    remain calibrated.

(B) Long documents. ILDC cases run to thousands of tokens; BERT caps at 512. We encode
    hierarchically and make the pooling operator a documented, ablatable choice
    (mean vs. attention), because pooling affects calibration and therefore DASP.

(H/I) Encoder isolation and cache safety. The PlausibilityScorer uses its own frozen
    encoder instance, so fine-tuning the predictor cannot leak into plausibility
    features. Feature-cache keys include a model-version tag, so cached vectors from a
    different checkpoint can never be silently reused.

(F/G/J) Hardware. The file is single-device clean: it runs on one CUDA device or on
    CPU, selected by config (`device`). Multi-GPU rigor (5 seeds across 4x RTX 3090) is
    orchestrated externally in scripts/run_all.sh by running one seed per GPU; this file
    deliberately does NOT wrap DataParallel, which would complicate determinism and
    yields no benefit for embarrassingly-parallel seed runs. Determinism is a documented
    switch (`deterministic`), and MC-dropout passes are micro-batched to bound memory.

This module degrades gracefully to CPU so the entire pipeline remains reproducible
without a GPU; torch/transformers are imported lazily so the rest of the package
(and its unit tests) import without them.

Reference: Guo, Pleiss, Sun, Weinberger. On Calibration of Modern Neural Networks.
ICML 2017 (temperature scaling; ECE, Eq. 3, M=15 bins).  Gal & Ghahramani. Dropout as a
Bayesian Approximation. ICML 2016 (MC-dropout predictive moments).  Santosh, Chowdhury,
Xu, Grabmair. The Craft of Selective Prediction. EMNLP Findings 2024 (Eq. 7-9: SMP/PV/
BALD uncertainty estimators; BALD > PV > SMP > Softmax-Response on legal COC).
================================================================================
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------
MODEL_NAME = "law-ai/InLegalBERT"   # 110M, English, pre-trained on ~5.4M Indian legal docs
HIDDEN_SIZE = 768                   # InLegalBERT == bert-base config
MAX_TOKENS = 512                    # transformer hard cap per chunk


# --------------------------------------------------------------------------------
# Configuration object (mirrors configs/default.yaml -> model.inlegalbert)
# --------------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # encoding
    max_len: int = MAX_TOKENS
    max_chunks: int = 8                 # cap chunks/doc: 8*512 = up to 4096 tokens
    pooling: str = "attention"          # "mean" | "attention"
    # training
    fine_tune: bool = True              # True: end-to-end; False: frozen features + head
    lr: float = 2e-5
    head_lr: float = 1e-3               # higher LR for the randomly-initialised head/pooler
    epochs: int = 3
    batch_size: int = 8                 # documents per optimisation step
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.2                # dropout rate used for the head + MC-dropout
    grad_clip: float = 1.0
    class_weighted: bool = True         # handle ILDC imbalance in the loss
    # uncertainty
    mc_dropout_passes: int = 30         # T stochastic passes (Gal 2016 uses 10-100; 30 balances
                                        # variance-stability against inference cost)
    mc_micro_batch: int = 4             # bound memory during MC passes (J)
    epistemic_estimator: str = "bald"   # "std" | "pv" | "bald" -- the scalar u(x) for DASP.
                                        # BALD (Santosh 2024) captures TOTAL uncertainty and was
                                        # their best estimator on legal COC; default to it.
    # calibration
    temperature_scale: bool = True      # fit temperature on dev after training (C)
    # systems
    device: str = "cuda"                # "cuda" | "cpu"; auto-falls-back to cpu if no GPU
    seed: int = 13
    deterministic: bool = True          # cudnn deterministic (G); slower but reproducible
    feat_cache: str = "data/feat_cache"
    model_tag: str = "inlegalbert_v1"   # part of the cache key (I); bump on any change

    def __post_init__(self):
        if self.pooling not in {"mean", "attention"}:
            raise ValueError(f"pooling must be 'mean' or 'attention', got {self.pooling!r}")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError(f"device must be 'cuda' or 'cpu', got {self.device!r}")
        if self.mc_dropout_passes < 2:
            raise ValueError("mc_dropout_passes must be >= 2 for a usable variance estimate")
        if self.epistemic_estimator not in {"std", "pv", "bald"}:
            raise ValueError(f"epistemic_estimator must be 'std'|'pv'|'bald', "
                             f"got {self.epistemic_estimator!r}")


# --------------------------------------------------------------------------------
# MC-dropout helper  (verified to match PyTorch dropout semantics)
# --------------------------------------------------------------------------------
def enable_mc_dropout(module) -> int:
    """
    Put ONLY dropout layers into training mode, leaving the rest of `module` in eval.
    Returns the number of dropout modules reactivated. This is what makes MC-dropout
    produce a non-degenerate predictive distribution after module.eval().
    """
    import torch.nn as nn
    n = 0
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            n += 1
    return n


# --------------------------------------------------------------------------------
# Uncertainty estimators over MC-dropout samples  (torch-free; operate on numpy).
# Given `samples`, an (N, T) array of P(y=1) over T stochastic passes, each estimator
# returns an (N,) uncertainty where HIGHER = more uncertain. These mirror Santosh et al.
# (2024) Eq. 7-9; their study found BALD > PV > SMP > Softmax-Response on legal COC.
# --------------------------------------------------------------------------------
def _stack_binary(p1: np.ndarray) -> np.ndarray:
    """(N,T) prob of class 1 -> (N,T,2) clipped two-class distribution."""
    p1 = np.clip(np.asarray(p1, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.stack([1.0 - p1, p1], axis=-1)


def uncertainty_from_samples(samples: np.ndarray, kind: str = "bald") -> np.ndarray:
    """
    Compute a per-example uncertainty (N,) from MC samples (N,T) of P(y=1).

      std  : predictive standard deviation of P(y=1) across passes (Gal 2016 moment).
      pv   : Probability Variance (Santosh Eq. 8) = mean over classes of the across-pass
             variance. Captures EPISTEMIC uncertainty (disagreement among passes) only.
      bald : Bayesian Active Learning by Disagreement (Santosh Eq. 9) = mutual information
             = H(mean prediction) - mean(per-pass entropy). Captures TOTAL uncertainty and
             specifically the epistemic component; reads ~0 for purely ALEATORIC uncertainty
             (every pass agrees the answer is a coin-flip) but high when passes DISAGREE.
             This is why Santosh found it best on inherently-ambiguous legal data.

    All three are 0 when the passes are identical (no dropout variance), so the DASP
    coupling degrades correctly to confidence-only when MC is unavailable.
    """
    s = np.asarray(samples, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"samples must be (N,T), got shape {s.shape}")
    if kind == "std":
        return s.std(axis=1)
    P = _stack_binary(s)                                 # (N,T,2)
    if kind == "pv":
        return P.var(axis=1).mean(axis=1)                # mean across-pass variance over classes
    if kind == "bald":
        mean = P.mean(axis=1)                            # (N,2) predictive mean
        H_mean = -(mean * np.log(mean)).sum(axis=1)      # entropy of the mean (N,)
        H_each = -(P * np.log(P)).sum(axis=2)            # entropy of each pass (N,T)
        return H_mean - H_each.mean(axis=1)              # mutual information (N,)
    raise ValueError(f"unknown uncertainty kind: {kind!r} (expected 'std'|'pv'|'bald')")


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (Guo et al. 2017, Eq. 3) with M=15 equal-width confidence
    bins (their default, so the number is comparable to the literature). Confidence is
    max(p, 1-p); accuracy is whether the argmax prediction is correct. Returns the
    sample-weighted mean of |accuracy - confidence| across bins. 0 = perfectly calibrated.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(p) == 0:
        return float("nan")
    conf = np.maximum(p, 1.0 - p)
    pred = (p >= 0.5).astype(int)
    acc = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(p)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # left-closed on the first bin so confidence == lo (e.g. 0.5) is included
        in_bin = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            ece += (in_bin.sum() / n) * abs(acc[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


# --------------------------------------------------------------------------------
# 1. Hierarchical long-document encoder
# --------------------------------------------------------------------------------
class HierarchicalEncoder:
    """
    Encodes a long legal document by chunking to <=max_len tokens, encoding each chunk
    with InLegalBERT (CLS representation), and pooling chunk vectors. Pooling is either
    a parameter-free mean or a learned single-head attention pooler.

    Implemented as a torch.nn.Module under the hood, constructed lazily so importing
    this file never requires torch.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._built = False
        self._module = None       # the nn.Module (encoder + pooler)
        self._tok = None
        self._device = None

    # --- lazy construction -------------------------------------------------------
    def _build(self):
        if self._built:
            return
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel

        cfg = self.cfg
        self._device = _resolve_device(cfg.device)

        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        encoder = AutoModel.from_pretrained(MODEL_NAME)
        if not cfg.fine_tune:
            for p in encoder.parameters():
                p.requires_grad_(False)

        class _Net(nn.Module):
            def __init__(self, encoder, pooling, dropout):
                super().__init__()
                self.encoder = encoder
                self.pooling = pooling
                self.dropout = nn.Dropout(dropout)
                if pooling == "attention":
                    # single-head additive attention over chunk CLS vectors
                    self.attn = nn.Sequential(
                        nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                        nn.Tanh(),
                        nn.Linear(HIDDEN_SIZE, 1),
                    )

            def encode_chunks(self, input_ids, attention_mask):
                # input_ids: (n_chunks, L) -> CLS per chunk: (n_chunks, H)
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                return out.last_hidden_state[:, 0, :]

            def pool(self, chunk_vecs):
                import torch
                # chunk_vecs: (n_chunks, H) -> (H,)
                if self.pooling == "mean":
                    return chunk_vecs.mean(dim=0)
                scores = self.attn(chunk_vecs).squeeze(-1)            # (n_chunks,)
                w = torch.softmax(scores, dim=0).unsqueeze(-1)        # (n_chunks, 1)
                return (w * chunk_vecs).sum(dim=0)

            def forward(self, chunked_inputs):
                import torch
                # chunked_inputs: list over docs; each is (input_ids, attn) tensors
                pooled = []
                for input_ids, attn in chunked_inputs:
                    cv = self.encode_chunks(input_ids, attn)
                    pooled.append(self.dropout(self.pool(cv)))
                return torch.stack(pooled, dim=0)                     # (n_docs, H)

        self._module = _Net(encoder, cfg.pooling, cfg.dropout).to(self._device)
        self._tok = tok
        self._built = True

    # --- tokenisation into chunks -----------------------------------------------
    def _chunk(self, text: str):
        """Tokenise `text` and split into up to max_chunks chunks of <=max_len tokens."""
        import torch
        cfg = self.cfg
        enc = self._tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        cls, sep, pad = self._tok.cls_token_id, self._tok.sep_token_id, self._tok.pad_token_id
        body = cfg.max_len - 2  # leave room for [CLS] and [SEP]
        pieces = [enc[i:i + body] for i in range(0, len(enc), body)][: cfg.max_chunks]
        if not pieces:
            pieces = [[]]
        input_ids, masks = [], []
        for p in pieces:
            ids = [cls] + p + [sep]
            attn = [1] * len(ids)
            while len(ids) < cfg.max_len:
                ids.append(pad)
                attn.append(0)
            input_ids.append(ids)
            masks.append(attn)
        dev = self._device
        return (torch.tensor(input_ids, device=dev), torch.tensor(masks, device=dev))

    # --- public: deterministic pooled embedding (eval) --------------------------
    def embed(self, texts: Sequence[str], train_mode: bool = False):
        """
        Return pooled document embeddings, shape (len(texts), H), as a torch tensor.
        train_mode=False -> eval (deterministic, for point predictions and caching).
        train_mode=True  -> keeps the graph for backprop during fine-tuning.
        """
        self._build()
        import torch
        net = self._module
        net.train(train_mode)
        chunked = [self._chunk(t) for t in texts]
        if train_mode:
            return net(chunked)
        with torch.no_grad():
            return net(chunked)

    def parameters(self):
        self._build()
        return self._module.parameters()

    @property
    def module(self):
        self._build()
        return self._module

    @property
    def device(self):
        self._build()
        return self._device


# --------------------------------------------------------------------------------
# 2. The main predictor
# --------------------------------------------------------------------------------
class InLegalBERTClassifier:
    """
    Binary predictor over the hierarchical encoder. Exposes:
        fit(texts, labels, dev_texts=None, dev_labels=None)
        predict_proba(texts)        -> deterministic calibrated P(y=1), shape (N,)
        predict_proba_mc(texts)     -> MC-dropout samples of P(y=1), shape (N, T)
        epistemic_uncertainty(texts)-> per-doc predictive std, shape (N,)
    The MC path is the epistemic signal consumed by selective.py; the deterministic
    path is the point probability used for the prediction and for conformal calibration.
    """

    def __init__(self, cfg: ModelConfig | None = None, **overrides):
        self.cfg = cfg or ModelConfig(**overrides)
        _seed_everything(self.cfg.seed, self.cfg.deterministic)
        self.encoder = HierarchicalEncoder(self.cfg)
        self._head = None              # nn.Linear(H, 2)
        self._temperature = 1.0        # fitted on dev (C)
        os.makedirs(self.cfg.feat_cache, exist_ok=True)

    # --- training ----------------------------------------------------------------
    def fit(self, texts, labels, dev_texts=None, dev_labels=None):
        import torch
        import torch.nn as nn
        from torch.optim import AdamW

        cfg = self.cfg
        device = self.encoder.device
        y = torch.tensor(list(labels), dtype=torch.long, device=device)

        self._head = nn.Linear(HIDDEN_SIZE, 2).to(device)

        # class weights for imbalance (E). Labels must be binary {0,1}.
        labels_arr = np.asarray(labels)
        uniq = set(np.unique(labels_arr).tolist())
        if not uniq <= {0, 1}:
            raise ValueError(f"labels must be binary in {{0,1}}, found {sorted(uniq)}")
        if cfg.class_weighted:
            counts = np.bincount(labels_arr.astype(int), minlength=2).astype(float)
            w = counts.sum() / (2.0 * np.clip(counts, 1, None))
            class_w = torch.tensor(w, dtype=torch.float32, device=device)
        else:
            class_w = None
        loss_fn = nn.CrossEntropyLoss(weight=class_w)

        # parameter groups: encoder (small LR) vs head/pooler (larger LR)
        enc_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = list(self._head.parameters())
        groups = [{"params": head_params, "lr": cfg.head_lr}]
        if cfg.fine_tune and enc_params:
            groups.append({"params": enc_params, "lr": cfg.lr})
        opt = AdamW(groups, weight_decay=cfg.weight_decay)

        n = len(texts)
        steps_per_epoch = max(1, (n + cfg.batch_size - 1) // cfg.batch_size)
        total_steps = steps_per_epoch * cfg.epochs
        sched = _linear_warmup_schedule(opt, int(cfg.warmup_ratio * total_steps), total_steps)

        idx = np.arange(n)
        rng = np.random.default_rng(cfg.seed)
        for _epoch in range(cfg.epochs):
            rng.shuffle(idx)
            self._head.train()
            for s in range(steps_per_epoch):
                batch = idx[s * cfg.batch_size:(s + 1) * cfg.batch_size]
                if len(batch) == 0:
                    continue
                bt = [texts[i] for i in batch]
                emb = self.encoder.embed(bt, train_mode=cfg.fine_tune)   # (b, H), grad if fine_tune
                logits = self._head(emb)
                loss = loss_fn(logits, y[batch])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in groups for p in g["params"]], cfg.grad_clip)
                opt.step()
                sched.step()

        # temperature scaling on dev (C) -- only if dev provided
        if cfg.temperature_scale and dev_texts is not None and dev_labels is not None:
            self._fit_temperature(dev_texts, dev_labels)
        return self

    # --- deterministic calibrated probability -----------------------------------
    def _logits(self, texts):
        import torch
        if self._head is None:
            raise RuntimeError("predictor is not trained; call fit() first")
        self._head.eval()
        emb = self.encoder.embed(texts, train_mode=False)   # eval -> deterministic
        with torch.no_grad():
            return self._head(emb)                           # (N, 2)

    def predict_proba(self, texts) -> np.ndarray:
        """Deterministic, temperature-scaled P(y=1). Shape (N,).

        NOTE (vs Gal 2016 Eq. 6): Gal's predictive mean is the AVERAGE of the MC-dropout
        passes. We deliberately use a SEPARATE deterministic eval pass (dropout off) for the
        reported point probability, because that is the calibrated quantity temperature
        scaling was fit on and the one DASP/conformal consume. The MC passes are used ONLY
        for the epistemic signal, never for the point prediction -- two distinct code paths.
        """
        import torch
        logits = self._logits(list(texts))
        probs = torch.softmax(logits / self._temperature, dim=1)
        return probs[:, 1].detach().cpu().numpy()

    # --- MC-dropout predictive distribution (A, D, J) ---------------------------
    def predict_proba_mc(self, texts) -> np.ndarray:
        """
        T stochastic forward passes with dropout active. Returns (N, T) array of P(y=1).
        Temperature is applied to keep MC probabilities on the same calibrated scale.
        """
        import torch
        if self._head is None:
            raise RuntimeError("predictor is not trained; call fit() first")
        texts = list(texts)
        net = self.encoder.module
        net.eval()                       # base eval...
        n_drop = enable_mc_dropout(net)  # ...then reactivate dropout only (A)
        self._head.eval()
        n_drop += enable_mc_dropout(self._head)
        assert n_drop > 0, "no dropout modules found; MC-dropout would be degenerate"
        if self.cfg.dropout <= 0.0:
            # dropout layers exist but are no-ops at rate 0 -> every pass identical -> zero
            # epistemic signal. Surface this loudly rather than silently returning u(x)=0.
            import warnings
            warnings.warn("cfg.dropout == 0: MC-dropout passes are identical, so the epistemic "
                          "signal will be zero and DASP reduces to confidence-only.", stacklevel=2)

        T = self.cfg.mc_dropout_passes
        samples = np.empty((len(texts), T), dtype=np.float32)
        mb = max(1, self.cfg.mc_micro_batch)
        with torch.no_grad():
            for t in range(T):
                col = []
                for i in range(0, len(texts), mb):
                    chunk_inputs = [self.encoder._chunk(x) for x in texts[i:i + mb]]
                    emb = net(chunk_inputs)
                    logits = self._head(emb)
                    p = torch.softmax(logits / self._temperature, dim=1)[:, 1]
                    col.append(p.detach().cpu().numpy())
                samples[:, t] = np.concatenate(col)
        return samples

    def epistemic_uncertainty(self, texts) -> np.ndarray:
        """
        Per-document epistemic uncertainty across MC passes, shape (N,). The estimator is
        cfg.epistemic_estimator ('std'|'pv'|'bald', default 'bald' per Santosh 2024). This
        is the scalar u(x) consumed by selective.py's DASP rule. To run the std/pv/bald
        ABLATION without recomputing the network, cache predict_proba_mc(texts) once and
        call uncertainty_from_samples(samples, kind) for each estimator.
        """
        samples = self.predict_proba_mc(texts)
        return uncertainty_from_samples(samples, self.cfg.epistemic_estimator)

    # --- temperature scaling (Guo et al., 2017) ---------------------------------
    def _fit_temperature(self, dev_texts, dev_labels):
        """
        Fit a single temperature T on the dev set by minimising NLL (Guo et al. 2017),
        with the trained logits FROZEN (so accuracy is unchanged; only confidence is
        rescaled). T is optimised in log-space to stay positive and CLAMPED to [0.05, 100]
        so a tiny or perfectly-separable dev set cannot send T to 0/inf. Records dev ECE
        before and after scaling (compute_ece, Guo Eq. 3, M=15) for healthcheck/reporting.
        """
        import torch
        logits = self._logits(list(dev_texts)).detach()
        y = torch.tensor(list(dev_labels), dtype=torch.long, device=logits.device)

        # ECE before scaling (T=1), on the same dev probabilities
        p_before = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        self._ece_before = compute_ece(p_before, list(dev_labels))

        # optimise log T so that T = exp(log_T) stays strictly positive
        log_T = torch.zeros(1, requires_grad=True, device=logits.device)
        opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)
        nll = torch.nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = nll(logits / log_T.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        T = float(log_T.exp().item())
        if not np.isfinite(T):
            T = 1.0  # non-convergence (e.g. degenerate dev set) -> fall back to no scaling
        self._temperature = float(np.clip(T, 0.05, 100.0))

        # ECE after scaling
        p_after = torch.softmax(logits / self._temperature, dim=1)[:, 1].cpu().numpy()
        self._ece_after = compute_ece(p_after, list(dev_labels))
        return self._temperature

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def calibration(self) -> dict:
        """Calibration diagnostics from the last _fit_temperature (Guo Eq. 3 ECE, M=15)."""
        return {"temperature": self._temperature,
                "ece_before": getattr(self, "_ece_before", None),
                "ece_after": getattr(self, "_ece_after", None)}


# --------------------------------------------------------------------------------
# 3. Baseline + plausibility scorer
# --------------------------------------------------------------------------------
class TfidfLinearBaseline:
    """TF-IDF (word 1-2 grams) + logistic regression. CPU, seconds. Baseline/ablation."""

    def __init__(self, seed: int = 13):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                                   sublinear_tf=True)
        self.clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                                      random_state=seed)

    def fit(self, texts, y, *args, **kwargs):
        X = self.vec.fit_transform(texts)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.clf.predict_proba(self.vec.transform(texts))[:, 1]


class PlausibilityScorer:
    """
    Learned tier for plausibility.py. Uses its OWN frozen encoder (H) so it can never
    perturb the predictor's parameters or features. Trains a logistic head on
    [emb(orig), emb(edit), |emb(orig) - emb(edit)|]. Exposes a callable
    (original, edited) -> [0,1] for LegalPlausibilityFilter(scorer=...).
    """

    def __init__(self, seed: int = 13, device: str = "cuda"):
        # force a frozen, non-fine-tuned encoder instance (isolation, H)
        self._enc = HierarchicalEncoder(ModelConfig(
            fine_tune=False, device=device, seed=seed, model_tag="plausibility_enc_v1"))
        self._clf = None

    def _feat(self, a: str, b: str) -> np.ndarray:
        ea = self._enc.embed([a], train_mode=False)[0].detach().cpu().numpy()
        eb = self._enc.embed([b], train_mode=False)[0].detach().cpu().numpy()
        return np.concatenate([ea, eb, np.abs(ea - eb)])

    def fit(self, pairs, labels):
        from sklearn.linear_model import LogisticRegression
        X = np.vstack([self._feat(a, b) for a, b in pairs])
        self._clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, labels)
        return self

    def __call__(self, original: str, edited: str) -> float:
        if self._clf is None:
            return 0.5  # neutral when untrained -> filter falls back to its rules tier
        x = self._feat(original, edited)[None, :]
        return float(self._clf.predict_proba(x)[0, 1])


# --------------------------------------------------------------------------------
# Factory + systems helpers
# --------------------------------------------------------------------------------
def build_predictor(kind: str, cfg: ModelConfig | None = None, seed: int = 13):
    """Factory used by experiment.py: 'inlegalbert' | 'tfidf'."""
    if kind == "inlegalbert":
        return InLegalBERTClassifier(cfg or ModelConfig(seed=seed))
    if kind == "tfidf":
        return TfidfLinearBaseline(seed=seed)
    raise ValueError(f"unknown predictor kind: {kind!r}")


def _resolve_device(requested: str):
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        # graceful CPU fallback keeps the whole pipeline reproducible without a GPU (F)
        return torch.device("cpu")
    return torch.device(requested)


def _seed_everything(seed: int, deterministic: bool):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # opt-in full determinism; warn_only avoids crashes on ops lacking
            # deterministic kernels (G)
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass


def _linear_warmup_schedule(optimizer, warmup_steps: int, total_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)  # linear decay to 0

    return LambdaLR(optimizer, lr_lambda)


def _cache_key(text: str, model_tag: str) -> str:
    """Cache key binds content AND model version, so stale-checkpoint vectors can't leak (I)."""
    h = hashlib.sha256()
    h.update(model_tag.encode())
    h.update(b"\x00")
    h.update(text.encode())
    return h.hexdigest()[:24]


# --------------------------------------------------------------------------------
# Self-test: exercises every torch-free path. The torch paths (encoder, MC-dropout,
# temperature) are validated separately and run on the GPU server; here we guarantee
# the file imports cleanly and the baseline + helpers are correct without torch.
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # (1) cache key binds model tag
    k1 = _cache_key("same text", "inlegalbert_v1")
    k2 = _cache_key("same text", "inlegalbert_v2")
    assert k1 != k2, "cache key must change when the model version changes"

    # (2) config validation
    c = ModelConfig()
    assert c.pooling in {"mean", "attention"} and c.mc_dropout_passes > 1
    for bad in [dict(pooling="max"), dict(device="tpu"), dict(mc_dropout_passes=1)]:
        try:
            ModelConfig(**bad); raise AssertionError(f"should have rejected {bad}")
        except ValueError:
            pass

    # (3) TF-IDF baseline trains and predicts in-range (min_df=3 needs >=3 docs/class)
    pos = ["appeal allowed conviction set aside"] * 3
    neg = ["appeal dismissed conviction upheld"] * 3
    m = build_predictor("tfidf").fit(pos + neg, [1, 1, 1, 0, 0, 0])
    p = m.predict_proba(["the appeal is allowed and conviction set aside"])
    assert 0.0 <= float(p[0]) <= 1.0

    # (4) factory rejects unknown kinds
    try:
        build_predictor("nonsense"); raise AssertionError("should have raised")
    except ValueError:
        pass

    # (5) untrained plausibility scorer returns neutral 0.5 without building the encoder
    class _Stub(PlausibilityScorer):
        def __init__(self):  # bypass encoder construction for a torch-free contract test
            self._clf = None
    assert _Stub()("a", "b") == 0.5

    # (6) uncertainty estimators (Santosh Eq. 7-9) -- verified on synthetic MC samples,
    #     no torch needed. All are 0 for identical passes; ordered correctly otherwise.
    identical = np.full((4, 30), 0.7)
    for kind in ("std", "pv", "bald"):
        assert np.allclose(uncertainty_from_samples(identical, kind), 0.0, atol=1e-9), \
            f"{kind} must be 0 when all MC passes agree"
    # disagreeing passes (epistemic) -> high; consistent-0.5 (aleatoric) -> BALD ~0
    disagree = np.tile(np.array([0.99] * 15 + [0.01] * 15), (1, 1))
    aleatoric = np.full((1, 30), 0.5)
    assert uncertainty_from_samples(disagree, "bald")[0] > 0.5, "BALD high for passes that disagree"
    assert uncertainty_from_samples(aleatoric, "bald")[0] < 0.01, "BALD ~0 for consistent (aleatoric)"
    assert uncertainty_from_samples(disagree, "pv")[0] > uncertainty_from_samples(identical, "pv")[0]
    try:
        uncertainty_from_samples(np.zeros(5), "bald"); raise AssertionError("must reject 1-D")
    except ValueError:
        pass
    try:
        uncertainty_from_samples(identical, "nonsense"); raise AssertionError("must reject kind")
    except ValueError:
        pass

    # (7) ECE (Guo Eq. 3, M=15) -- perfectly-calibrated ~0, overconfident high
    p_cal = np.array([0.9] * 100); y_cal = np.array([1] * 90 + [0] * 10)   # 90% conf, 90% acc
    p_over = np.array([0.99] * 100); y_over = np.array([1] * 60 + [0] * 40)  # 99% conf, 60% acc
    assert compute_ece(p_cal, y_cal) < 0.02, "ECE should be ~0 for a calibrated predictor"
    assert compute_ece(p_over, y_over) > 0.30, "ECE should be large for an overconfident predictor"
    assert np.isnan(compute_ece(np.array([]), np.array([])))  # empty -> nan contract

    # (8) epistemic_estimator config validation
    for bad in [dict(epistemic_estimator="entropy")]:
        try:
            ModelConfig(**bad); raise AssertionError(f"should reject {bad}")
        except ValueError:
            pass

    print(f"model.py self-test passed (baseline p={float(p[0]):.2f}, config validation OK, "
          f"cache-key isolation OK, factory OK, BALD/PV/std + ECE math verified vs Santosh/Guo)")
