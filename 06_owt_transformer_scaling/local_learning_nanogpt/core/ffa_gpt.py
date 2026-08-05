"""
FFA-GPT: Forward-Forward Algorithm for Transformer Language Models.
Based on nanoGPT architecture. Each transformer block has a local FFA loss.

Two goodness variants:
  - FFA-norm: G(h) = ||h||^2, pos = real sequence, neg = token-shuffled sequence
  - FFA-NCE:  G(h, y) = h @ e_y, pos = true next token, neg = K random tokens

Usage:
  python ffa_gpt.py --mode bp       # BP baseline
  python ffa_gpt.py --mode ffa_norm # FFA with norm goodness
  python ffa_gpt.py --mode ffa_nce  # FFA with NCE goodness
"""

import os
import sys
import math
import time
import json
import pickle
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Model components (reused from nanoGPT)
# ============================================================

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# ============================================================
# FFA Goodness Heads
# ============================================================

class NormGoodnessHead(nn.Module):
    """G(h) = ||h||^2 / d. Positive = real data, Negative = shuffled tokens."""
    def __init__(self, config):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(config.n_embd)))  # threshold ≈ d

    def goodness(self, h):
        # h: (B, T, d) -> scalar per position
        return (h * h).sum(dim=-1) / h.size(-1)  # (B, T)

    def loss(self, h_pos, h_neg):
        g_pos = self.goodness(h_pos)  # (B, T)
        g_neg = self.goodness(h_neg)  # (B, T)
        loss_pos = -F.logsigmoid(g_pos - self.theta).mean()
        loss_neg = -F.logsigmoid(self.theta - g_neg).mean()
        return loss_pos + loss_neg


class SymBaGoodnessHead(nn.Module):
    """SymBa loss (Lee & Song 2023): symmetric goodness loss.
    L = softplus(-alpha * (G_pos - G_neg)), G(h) = ||h||^2 / d.
    Fixes gradient imbalance of original FFA: both pos/neg gradients vanish
    simultaneously when Delta = G_pos - G_neg → +inf.
    Positive = real token sequence; Negative = token-shuffled sequence.
    """
    def __init__(self, config, alpha=4.0):
        super().__init__()
        self.alpha = alpha

    def goodness(self, h):
        return (h * h).sum(dim=-1) / h.size(-1)  # (B, T)

    def loss(self, h_pos, h_neg):
        delta = self.goodness(h_pos) - self.goodness(h_neg)  # (B, T)
        return F.softplus(-self.alpha * delta).mean()


class NCEGoodnessHead(nn.Module):
    """G(h, y) = h @ e_y. Positive = true next token, Negative = K random tokens."""
    def __init__(self, config, embedding_weight):
        super().__init__()
        self._embedding_weight_ref = [embedding_weight]  # shared with wte: (vocab_size, d)
        self.n_neg = config.n_neg  # number of negative samples
        self.theta = nn.Parameter(torch.tensor(0.0))  # threshold

    @property
    def embedding_weight(self):
        return self._embedding_weight_ref[0]

    def goodness(self, h, token_ids):
        # h: (B, T, d), token_ids: (B, T) -> (B, T)
        emb = self.embedding_weight[token_ids]  # (B, T, d)
        return (h * emb).sum(dim=-1)  # dot product per position

    def loss(self, h, targets):
        # h: (B, T, d), targets: (B, T) — next token ids
        B, T, d = h.size()

        # Positive: true next token
        g_pos = self.goodness(h, targets)  # (B, T)
        loss_pos = -F.logsigmoid(g_pos - self.theta).mean()

        # Negative: K random tokens
        neg_ids = torch.randint(0, self.embedding_weight.size(0),
                                (B, T, self.n_neg), device=h.device)
        neg_emb = self.embedding_weight[neg_ids]  # (B, T, K, d)
        g_neg = (h.unsqueeze(2) * neg_emb).sum(dim=-1)  # (B, T, K)
        loss_neg = -F.logsigmoid(self.theta - g_neg).mean()

        return loss_pos + loss_neg


class InBatchNCEGoodnessHead(nn.Module):
    """G(h, y) = h @ e_y with IN-BATCH negatives.
    Uses ALL other tokens in the batch as negatives → ~B*T negatives per position.
    For vocab=50257, batch (32,256) → ~5000 unique neg tokens → 10% coverage.
    Zero extra embedding lookup cost — just one matmul.
    """
    def __init__(self, config, embedding_weight):
        super().__init__()
        self._embedding_weight_ref = [embedding_weight]
        self.theta = nn.Parameter(torch.tensor(0.0))

    @property
    def embedding_weight(self):
        return self._embedding_weight_ref[0]

    def loss(self, h, targets):
        # h: (B, T, d), targets: (B, T)
        B, T, d = h.size()
        N = B * T

        h_flat = h.reshape(N, d)
        tgt_flat = targets.reshape(N)

        # Unique targets in this batch
        unique_tgts, inverse = torch.unique(tgt_flat, return_inverse=True)
        U = unique_tgts.size(0)
        tgt_emb = self.embedding_weight[unique_tgts]  # (U, d)

        # All-pairs scores: (N, U)
        scores = h_flat @ tgt_emb.T

        # Positive: each position's true target
        pos_idx = inverse  # (N,)
        g_pos = scores[torch.arange(N, device=h.device), pos_idx]  # (N,)
        loss_pos = -F.logsigmoid(g_pos - self.theta).mean()

        # Negative: per-position mean (balanced 1:1 with positive)
        # Set positive entries to a large value so logsigmoid(theta - large) ≈ 0
        neg_scores = scores.clone()
        neg_scores[torch.arange(N, device=h.device), pos_idx] = -1e9  # mask out pos
        # Per-position mean of negative losses: (N, U) → mean over U per row → mean over N
        neg_loss_all = -F.logsigmoid(self.theta - neg_scores)  # (N, U)
        neg_loss_all[torch.arange(N, device=h.device), pos_idx] = 0.0  # zero out masked
        loss_neg = neg_loss_all.sum(dim=1) / (U - 1)  # per-position mean
        loss_neg = loss_neg.mean()  # mean over positions

        return loss_pos + loss_neg


# ============================================================
class MSPGoodnessHead(nn.Module):
    """Multi-Scale Predictive (MSP) goodness head.

    Layer ℓ predicts the token K_ℓ steps ahead: G = h · e_{x_{t+K_ℓ}}.
    K_ℓ = min(2^layer_idx, K_max) — shallow layers look 1 step ahead,
    deep layers look further, forcing each layer to carry distinct information.
    This directly combats the information-decay bottleneck (Prop prop:info_decay).

    The goodness_head.loss() signature differs from NCEGoodnessHead:
    it takes x_full (B, T+K_max) instead of targets (B, T), so that
    any layer can look up its own K_ℓ-step targets.
    """
    def __init__(self, config, embedding_weight, layer_idx, n_neg=128, K_max=64):
        super().__init__()
        self._embedding_weight_ref = [embedding_weight]
        self.K = min(2 ** layer_idx, K_max)  # horizon for this layer
        self.n_neg = n_neg
        self.theta = nn.Parameter(torch.tensor(0.0))
        # Per-layer projection to the K_ℓ-step prediction subspace (d→d, no bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        nn.init.eye_(self.proj.weight)  # start as identity; fine-tuned toward future

    @property
    def embedding_weight(self):
        return self._embedding_weight_ref[0]

    def loss(self, h, x_full):
        """
        h:      (B, T, d)       — current layer output (block_size = T)
        x_full: (B, T+K_max)    — full token sequence including future tokens
        """
        B, T, d = h.shape
        K = self.K

        # K-step targets: x_full[:, K : K+T]
        targets = x_full[:, K: K + T]          # (B, T)

        h_proj = self.proj(h)                   # (B, T, d) — layer-specific direction

        target_emb = self.embedding_weight[targets]          # (B, T, d)
        g_pos = (h_proj * target_emb).sum(dim=-1)            # (B, T)
        loss_pos = -F.logsigmoid(g_pos - self.theta).mean()

        neg_ids = torch.randint(0, self.embedding_weight.size(0),
                                (B, T, self.n_neg), device=h.device)
        neg_emb = self.embedding_weight[neg_ids]             # (B, T, n_neg, d)
        g_neg = (h_proj.unsqueeze(2) * neg_emb).sum(dim=-1) # (B, T, n_neg)
        loss_neg = -F.logsigmoid(self.theta - g_neg).mean()

        return loss_pos + loss_neg


# ============================================================
class LocalCEHead(nn.Module):
    """Local cross-entropy head: projects h → vocab logits → CE loss.
    Equivalent to Mono-Forward / CaFo approach. NOT contrastive — uses full softmax.
    Shares embedding weight with wte (weight tying) for parameter efficiency.
    """
    def __init__(self, config, embedding_weight):
        super().__init__()
        self._embedding_weight_ref = [embedding_weight]  # (vocab_size, d) — shared with wte
        # No extra parameters needed due to weight tying: logits = h @ W_e^T

    @property
    def embedding_weight(self):
        return self._embedding_weight_ref[0]

    def logits(self, h):
        return h @ self.embedding_weight.T

    def loss(self, h, targets):
        # h: (B, T, d), targets: (B, T)
        logits = self.logits(h)  # (B, T, vocab_size) — weight-tied
        B, T, V = logits.size()
        return F.cross_entropy(logits.view(B*T, V), targets.view(B*T))


class DetachedLocalCEHead(nn.Module):
    """Local CE head using a detached shared token embedding as the decoder.

    This is the zero-extra-parameter version of local CE. It keeps the full-vocab
    softmax signal while preventing local CE losses from updating wte through the
    decoder branch. Gradients can still reach wte through the input embedding path
    for the first local block.
    """
    def __init__(self, embedding_weight):
        super().__init__()
        self._embedding_weight_ref = [embedding_weight]

    @property
    def embedding_weight(self):
        return self._embedding_weight_ref[0]

    def logits(self, h):
        return h @ self.embedding_weight.detach().T

    def loss(self, h, targets):
        logits = self.logits(h)
        B, T, V = logits.size()
        return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))


class ProjectedLocalCEHead(nn.Module):
    """Local CE head with a per-layer hidden projection and detached decoder."""
    def __init__(self, n_embd, embedding_weight):
        super().__init__()
        self._embedding_weight_ref = [embedding_weight]
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        nn.init.eye_(self.proj.weight)

    @property
    def embedding_weight(self):
        return self._embedding_weight_ref[0]

    def logits(self, h):
        projected = self.proj(h)
        return projected @ self.embedding_weight.detach().T

    def loss(self, h, targets):
        logits = self.logits(h)
        B, T, V = logits.size()
        return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))


class LowRankVocabAdapterCEHead(nn.Module):
    """Local CE head with W_l = stopgrad(E) + A_l B_l.

    The low-rank adapter gives each layer an independent vocabulary correction
    without storing a full V x d local classifier.
    """
    def __init__(self, n_embd, vocab_size, embedding_weight, rank):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive for LowRankVocabAdapterCEHead.")
        self._embedding_weight_ref = [embedding_weight]
        self.rank = min(rank, n_embd, vocab_size)
        self.A = nn.Parameter(torch.empty(vocab_size, self.rank))
        self.B = nn.Parameter(torch.zeros(self.rank, n_embd))
        nn.init.normal_(self.A, std=0.02)

    @property
    def embedding_weight(self):
        return self._embedding_weight_ref[0]

    def logits(self, h):
        base_logits = h @ self.embedding_weight.detach().T
        adapter_logits = (h @ self.B.T) @ self.A.T
        return base_logits + adapter_logits

    def loss(self, h, targets):
        logits = self.logits(h)
        B, T, V = logits.size()
        return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))


class UntiedLocalCEHead(nn.Module):
    """Per-layer CE head with INDEPENDENT weights (NOT shared with wte).

    Fixes the gradient conflict in LocalCEHead (lce2 failure):
      - LocalCEHead: logits = h @ wte^T  → shared wte receives conflicting gradients
      - UntiedLocalCEHead: logits = h @ W_ℓ^T  → W_ℓ per-layer, no wte conflict

    Since δ_neg = 0 (full vocab softmax), recovers BP-quality gradient signal
    at each layer WITHOUT using shared wte — solving both:
      (a) gradient conflict: wte not pulled by multiple objectives
      (b) trivial negatives: δ_neg=0 → all negatives are informative

    Memory cost: L × vocab_size × n_embd extra parameters.
    For large (L=8, d=512, V=50K): ~205M extra params (GPU has 97GB, well within limit).
    """
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        nn.init.normal_(self.head.weight, std=0.02)

    def logits(self, h):
        return self.head(h)

    def loss(self, h, targets):
        # h: (B, T, d), targets: (B, T)
        logits = self.logits(h)  # (B, T, V) — independent weights, no wte sharing
        B, T, V = logits.size()
        return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))


# ============================================================
# FFA Transformer Block
# ============================================================

class FFABlock(nn.Module):
    """Transformer block with local loss. No gradient flows across blocks."""
    def __init__(self, config, goodness_type='norm', embedding_weight=None,
                 layer_idx=0, K_max=64):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        if goodness_type == 'norm':
            self.goodness_head = NormGoodnessHead(config)
        elif goodness_type == 'symba':
            self.goodness_head = SymBaGoodnessHead(config)
        elif goodness_type == 'nce':
            self.goodness_head = NCEGoodnessHead(config, embedding_weight)
        elif goodness_type == 'inbatch':
            self.goodness_head = InBatchNCEGoodnessHead(config, embedding_weight)
        elif goodness_type == 'local_ce':
            self.goodness_head = LocalCEHead(config, embedding_weight)
        elif goodness_type == 'detached_local_ce':
            self.goodness_head = DetachedLocalCEHead(embedding_weight)
        elif goodness_type == 'projected_local_ce':
            self.goodness_head = ProjectedLocalCEHead(config.n_embd, embedding_weight)
        elif goodness_type == 'lowrank_vocab_adapter_ce':
            self.goodness_head = LowRankVocabAdapterCEHead(
                config.n_embd,
                config.vocab_size,
                embedding_weight,
                config.lce_rank,
            )
        elif goodness_type == 'untied_ce':
            self.goodness_head = UntiedLocalCEHead(config.n_embd, config.vocab_size)
        elif goodness_type == 'msp':
            self.goodness_head = MSPGoodnessHead(
                config, embedding_weight, layer_idx=layer_idx,
                n_neg=config.n_neg, K_max=K_max)
        self.goodness_type = goodness_type

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ============================================================
# FFA-GPT Model
# ============================================================

@dataclass
class FFAGPTConfig:
    block_size: int = 256
    vocab_size: int = 65  # Shakespeare char
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False
    n_neg: int = 16  # NCE negative samples
    lce_rank: int = 64  # rank for low-rank local CE vocabulary adapters


class FFAGPT(nn.Module):
    def __init__(self, config, mode='ffa_norm'):
        super().__init__()
        self.config = config
        self.mode = mode

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        goodness_type = {'ffa_norm': 'norm', 'ffa_symba': 'symba',
                         'ffa_nce': 'nce', 'ffa_inbatch': 'inbatch',
                         'ffa_local_ce': 'local_ce', 'ffa_msp': 'msp',
                         'ffa_lce_untied': 'untied_ce',
                         'ffa_pd_lc': 'detached_local_ce',
                         'ffa_plce': 'projected_local_ce',
                         'ffa_lrva_lce': 'lowrank_vocab_adapter_ce'}.get(mode, 'nce')
        # K_max for MSP: cap horizon at block_size // 4 to avoid sequence boundary issues
        K_max = max(1, config.block_size // 4)
        self.K_max = K_max
        self.blocks = nn.ModuleList([
            FFABlock(config, goodness_type=goodness_type,
                     embedding_weight=self.wte.weight,
                     layer_idx=i, K_max=K_max)
            for i in range(config.n_layer)
        ])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # For BP baseline and evaluation
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_embeddings(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        return self.drop(tok_emb + pos_emb)

    def forward_bp(self, idx, targets=None):
        """Standard BP forward pass (baseline)."""
        x = self.get_embeddings(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def forward_ffa_norm(self, idx_pos, idx_neg):
        """FFA-norm: two forward passes with pos/neg data. Returns per-layer losses."""
        x_pos = self.get_embeddings(idx_pos)
        x_neg = self.get_embeddings(idx_neg)

        layer_losses = []
        for block in self.blocks:
            # Detach inputs: no gradient flows across blocks
            x_pos_in = x_pos.detach().requires_grad_(True)
            x_neg_in = x_neg.detach().requires_grad_(True)

            x_pos = block(x_pos_in)
            x_neg = block(x_neg_in)

            loss = block.goodness_head.loss(x_pos, x_neg)
            layer_losses.append(loss)

        return layer_losses

    def forward_ffa_symba(self, idx_pos, idx_neg):
        """SymBa: symmetric goodness loss, same pos/neg protocol as ffa_norm.
        Identical to forward_ffa_norm() but calls SymBaGoodnessHead.loss().
        """
        x_pos = self.get_embeddings(idx_pos)
        x_neg = self.get_embeddings(idx_neg)
        layer_losses = []
        for block in self.blocks:
            x_pos_in = x_pos.detach().requires_grad_(True)
            x_neg_in = x_neg.detach().requires_grad_(True)
            x_pos = block(x_pos_in)
            x_neg = block(x_neg_in)
            loss = block.goodness_head.loss(x_pos, x_neg)
            layer_losses.append(loss)
        return layer_losses

    def forward_ffa_lce_olu(self, idx, targets, group_size=2):
        """OLU (Overlapping Local Updates): train adjacent groups of blocks jointly.
        Within each group, gradients flow freely between all blocks in the group.
        Between groups, inputs are detached (no cross-group gradient).

        group_size=2 → pairs (0,1), (2,3), ...
        Returns list of one summed loss per group.
        """
        x = self.get_embeddings(idx)
        group_losses = []
        i = 0
        while i < len(self.blocks):
            j = min(i + group_size, len(self.blocks))
            # Detach from previous group; first group can propagate to embeddings
            x_in = x if i == 0 else x.detach().requires_grad_(True)
            x_curr = x_in
            group_loss = None
            for k in range(i, j):
                x_curr = self.blocks[k](x_curr)
                bloss = self.blocks[k].goodness_head.loss(x_curr, targets)
                group_loss = bloss if group_loss is None else group_loss + bloss
            group_losses.append(group_loss)
            x = x_curr
            i = j
        return group_losses

    def forward_ffa_nce(self, idx, targets):
        """FFA-NCE / FFA-InBatch: one forward pass, local loss at each layer."""
        x = self.get_embeddings(idx)

        layer_losses = []
        for block in self.blocks:
            x_in = x.detach().requires_grad_(True)
            x = block(x_in)
            loss = block.goodness_head.loss(x, targets)
            layer_losses.append(loss)

        return layer_losses

    def forward_ffa_msp(self, x_full):
        """MSP forward pass.

        x_full: (B, block_size + K_max) — input tokens PLUS K_max future tokens.
        Each layer's MSPGoodnessHead looks up its own K_ℓ-step targets from x_full.
        Returns list of per-layer losses.
        """
        T = self.config.block_size
        x = self.get_embeddings(x_full[:, :T])   # (B, T, d) — embed only input portion

        layer_losses = []
        for block in self.blocks:
            x_in = x.detach().requires_grad_(True)
            x = block(x_in)
            loss = block.goodness_head.loss(x, x_full)  # MSPGoodnessHead uses x_full
            layer_losses.append(loss)

        return layer_losses

    def _forward_local_ce(self, idx, targets):
        """Run a local CE variant with no gradient flow across blocks.

        Block 0: gradient allowed to flow to wte/wpe.
          - In NCE mode, wte receives gradient via the explicit e_y embedding lookup in the loss.
          - In detached-decoder CE variants, wte is NOT updated through the decoder branch,
            so we skip the detach on block 0 to let its CE gradient propagate back to wte
            through the forward embeddings.
        Blocks 1+: detach (greedy local learning, no gradient to earlier blocks).

        Returns list of per-layer CE losses.
        """
        x = self.get_embeddings(idx)

        layer_losses = []
        for i, block in enumerate(self.blocks):
            if i == 0:
                x_in = x  # no detach: gradient flows to wte/wpe from block 0's CE loss
            else:
                x_in = x.detach().requires_grad_(True)
            x = block(x_in)
            loss = block.goodness_head.loss(x, targets)
            layer_losses.append(loss)

        return layer_losses

    def forward_ffa_lce_untied(self, idx, targets):
        """FFA with per-layer untied CE heads."""
        return self._forward_local_ce(idx, targets)

    def forward_ffa_pd_lc(self, idx, targets):
        """FFA with detached shared local CE decoder and no extra head parameters."""
        return self._forward_local_ce(idx, targets)

    def forward_ffa_plce(self, idx, targets):
        """FFA with per-layer projected local CE and detached shared decoder."""
        return self._forward_local_ce(idx, targets)

    def forward_ffa_lrva_lce(self, idx, targets):
        """FFA with low-rank vocabulary-adapter local CE heads."""
        return self._forward_local_ce(idx, targets)

    def evaluate_perplexity(self, idx, targets):
        """Evaluate perplexity.

        - bp / ffa_nce / ffa_msp: use shared lm_head (weight-tied with wte), with ln_f.
        - ffa_lce_untied: use last block's untied CE head directly (no ln_f,
          consistent with how it was trained — ln_f was not part of the local CE loss).
        """
        with torch.no_grad():
            x = self.get_embeddings(idx)
            for block in self.blocks:
                x = block(x)
            if self.mode in {'ffa_lce_untied', 'ffa_pd_lc', 'ffa_plce', 'ffa_lrva_lce'}:
                logits = self.blocks[-1].goodness_head.logits(x)
            else:
                x = self.ln_f(x)
                logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss.item()

    def compute_gram_erank(self, idx):
        """Compute Gram effective rank at each layer."""
        with torch.no_grad():
            x = self.get_embeddings(idx)
            eranks = []
            for block in self.blocks:
                x = block(x)
                # h: (B, T, d) -> flatten to (B*T, d)
                h = x.view(-1, x.size(-1))
                # Gram: H = h^T h / n
                H = h.T @ h / h.size(0)
                eigs = torch.linalg.eigvalsh(H)
                eigs = eigs[eigs > 1e-10]
                p = eigs / eigs.sum()
                entropy = -(p * torch.log(p)).sum()
                eranks.append(float(torch.exp(entropy)))
            return eranks


# ============================================================
# Data loading (from nanoGPT)
# ============================================================

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_batch_msp(data, block_size, batch_size, K_max, device):
    """Get a batch with K_max extra future tokens for MSP multi-horizon prediction.

    Returns x_full of shape (B, block_size + K_max), where:
      x_full[:, :block_size]  — input tokens
      x_full[:, k:k+block_size] — k-step targets (k=1..K_max)
    """
    T_total = block_size + K_max
    ix = torch.randint(len(data) - T_total, (batch_size,))
    x_full = torch.stack([torch.from_numpy(data[i: i + T_total].astype(np.int64)) for i in ix])
    return x_full.to(device)


def make_negative_batch(x):
    """Create negative samples by shuffling tokens within each sequence."""
    B, T = x.size()
    x_neg = x.clone()
    for i in range(B):
        perm = torch.randperm(T, device=x.device)
        x_neg[i] = x[i, perm]
    return x_neg


# ============================================================
# Training
# ============================================================

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}, Mode: {args.mode}")

    # Load data
    data_dir = args.data_dir if args.data_dir else os.path.join(args.nanogpt_dir, 'data', 'shakespeare_char')
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    # Load meta
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"Vocab size: {vocab_size}, Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

    # Model config
    config = FFAGPTConfig(
        block_size=args.block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        n_neg=args.n_neg,
    )
    model = FFAGPT(config, mode=args.mode).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.2f}M")

    # Optimizer setup
    if args.mode == 'bp':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    else:
        # Fix #1: separate optimizer for embeddings (wte, wpe) — they participate in
        # NCE goodness (G = h @ e_y) but were previously NOT updated.
        optimizers = []
        for block in model.blocks:
            optimizers.append(torch.optim.AdamW(block.parameters(), lr=args.lr, weight_decay=0.1))
        # Embedding optimizer: updates wte and wpe via gradients from all layers' NCE losses
        emb_params = list(model.wte.parameters()) + list(model.wpe.parameters())
        emb_optimizer = torch.optim.AdamW(emb_params, lr=args.lr, weight_decay=0.01)

    # LR schedule: warmup + cosine decay
    def get_lr(step):
        warmup_iters = args.warmup_iters
        if step < warmup_iters:
            return args.lr * step / warmup_iters
        decay_ratio = (step - warmup_iters) / max(1, args.max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.lr - args.min_lr)

    # Training loop
    results = {
        'mode': args.mode,
        'config': {k: v for k, v in vars(config).items()},
        'train_losses': [],
        'val_perplexities': [],
        'layer_losses': [],
        'gram_eranks': [],
        'wall_times': [],
    }

    # For MSP, we need extra future tokens in each batch
    K_max = model.K_max if args.mode == 'ffa_msp' else 0

    t0 = time.time()
    for step in range(args.max_iters):
        model.train()
        if args.mode == 'ffa_msp':
            x_full = get_batch_msp(train_data, config.block_size, args.batch_size, K_max, device)
            x = x_full[:, :config.block_size]
            y = x_full[:, 1: config.block_size + 1]  # 1-step targets (for eval reference)
        else:
            x, y = get_batch(train_data, config.block_size, args.batch_size, device)

        # Update learning rate (cosine schedule)
        lr = get_lr(step)
        if args.mode == 'bp':
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            for opt in optimizers:
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
            for param_group in emb_optimizer.param_groups:
                param_group['lr'] = lr

        if args.mode == 'bp':
            _, loss = model.forward_bp(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss = loss.item()

        elif args.mode == 'ffa_norm':
            x_neg = make_negative_batch(x)
            layer_losses = model.forward_ffa_norm(x, x_neg)
            train_loss = 0
            emb_optimizer.zero_grad()
            for opt, ll in zip(optimizers, layer_losses):
                opt.zero_grad()
                ll.backward()
                torch.nn.utils.clip_grad_norm_(list(opt.param_groups[0]['params']), 1.0)
                opt.step()
                train_loss += ll.item()
            # Fix #1: update embeddings with accumulated gradients from all layers
            torch.nn.utils.clip_grad_norm_(emb_params, 1.0)
            emb_optimizer.step()
            train_loss /= len(layer_losses)

        elif args.mode == 'ffa_msp':
            layer_losses = model.forward_ffa_msp(x_full)
            train_loss = 0
            emb_optimizer.zero_grad()
            for opt, ll in zip(optimizers, layer_losses):
                opt.zero_grad()
                ll.backward()
                torch.nn.utils.clip_grad_norm_(list(opt.param_groups[0]['params']), 1.0)
                opt.step()
                train_loss += ll.item()
            torch.nn.utils.clip_grad_norm_(emb_params, 1.0)
            emb_optimizer.step()
            train_loss /= len(layer_losses)

        elif args.mode in ('ffa_nce', 'ffa_inbatch', 'ffa_local_ce'):
            layer_losses = model.forward_ffa_nce(x, y)

            if args.mode == 'ffa_local_ce':
                # LocalCE: shared embedding causes version conflict with sequential backward.
                # Fix: sum all losses → single backward → local grads (detach ensures locality).
                emb_optimizer.zero_grad()
                for opt in optimizers:
                    opt.zero_grad()
                total_loss = sum(layer_losses)
                total_loss.backward()
                for opt in optimizers:
                    torch.nn.utils.clip_grad_norm_(list(opt.param_groups[0]['params']), 1.0)
                    opt.step()
                torch.nn.utils.clip_grad_norm_(emb_params, 1.0)
                emb_optimizer.step()
                train_loss = total_loss.item() / len(layer_losses)
            else:
                # NCE / InBatch: sequential backward (no shared param conflict)
                train_loss = 0
                emb_optimizer.zero_grad()
                for opt, ll in zip(optimizers, layer_losses):
                    opt.zero_grad()
                    ll.backward()
                    torch.nn.utils.clip_grad_norm_(list(opt.param_groups[0]['params']), 1.0)
                    opt.step()
                    train_loss += ll.item()
                torch.nn.utils.clip_grad_norm_(emb_params, 1.0)
                emb_optimizer.step()
                train_loss /= len(layer_losses)

        results['train_losses'].append(train_loss)

        # Evaluate periodically
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            model.eval()
            val_x, val_y = get_batch(val_data, config.block_size, args.batch_size, device)
            val_loss = model.evaluate_perplexity(val_x, val_y)
            val_ppl = math.exp(min(val_loss, 20))  # cap at exp(20) to avoid overflow

            eranks = model.compute_gram_erank(val_x)
            results['val_perplexities'].append({'step': step, 'ppl': val_ppl, 'loss': val_loss})
            results['gram_eranks'].append({'step': step, 'eranks': eranks})

            elapsed = time.time() - t0
            results['wall_times'].append({'step': step, 'time': elapsed})
            print(f"Step {step:5d} | train_loss {train_loss:.4f} | val_ppl {val_ppl:.2f} | "
                  f"erank {[f'{e:.1f}' for e in eranks]} | {elapsed:.1f}s")

    # Phase 2: Fine-tune lm_head on top of frozen FFA features (linear probe)
    if args.mode.startswith('ffa'):
        print("\n--- Fine-tuning lm_head (linear probe) ---")
        # Freeze all blocks, only train lm_head (which shares weight with wte)
        for block in model.blocks:
            for p in block.parameters():
                p.requires_grad = False
        for p in model.wpe.parameters():
            p.requires_grad = False
        # lm_head.weight = wte.weight (tied), so training lm_head also updates wte
        probe_optimizer = torch.optim.AdamW(model.lm_head.parameters(), lr=args.lr, weight_decay=0.01)
        for probe_step in range(args.probe_iters):
            model.train()
            px, py = get_batch(train_data, config.block_size, args.batch_size, device)
            _, probe_loss = model.forward_bp(px, py)
            probe_optimizer.zero_grad()
            probe_loss.backward()
            probe_optimizer.step()
            if probe_step % 100 == 0:
                model.eval()
                vx, vy = get_batch(val_data, config.block_size, args.batch_size, device)
                vl = model.evaluate_perplexity(vx, vy)
                vp = math.exp(min(vl, 20))
                print(f"  Probe step {probe_step:4d} | val_ppl {vp:.2f}")
                results['val_perplexities'].append({'step': args.max_iters + probe_step, 'ppl': vp, 'loss': vl, 'phase': 'probe'})
        # Unfreeze for potential further use
        for block in model.blocks:
            for p in block.parameters():
                p.requires_grad = True

    # Save results
    os.makedirs(os.path.join(args.output_dir, 'results'), exist_ok=True)
    out_path = os.path.join(args.output_dir, 'results', f'ffa_lm_{args.mode}_{args.tag}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='bp', choices=['bp', 'ffa_norm', 'ffa_nce', 'ffa_inbatch', 'ffa_local_ce', 'ffa_msp', 'ffa_lce_untied'])
    parser.add_argument('--nanogpt_dir', type=str, default='/home/zw868/Desktop/nanoGPT')
    parser.add_argument('--data_dir', type=str, default='', help='Override data directory (default: nanogpt_dir/data/shakespeare_char)')
    parser.add_argument('--output_dir', type=str, default='/home/zw868/Desktop/FFA-theory')
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--n_embd', type=int, default=384)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--n_neg', type=int, default=64)
    parser.add_argument('--probe_iters', type=int, default=500)
    parser.add_argument('--tag', type=str, default='v2')
    args = parser.parse_args()
    train(args)
