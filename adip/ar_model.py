"""Causal autoregressive Transformer over binary decision variables."""

from __future__ import annotations

import math
import pathlib
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Float mask added to attention scores: (L, L), -inf above diagonal."""
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32),
        diagonal=1,
    )


def sparse_attn_window_tokens(num_sites: int, sparse_attn_prev_fraction: float) -> Optional[int]:
    """
    Sliding causal window length: each query attends to at most this many **keys**
    ending at the query position (including itself).

    ``None`` means full dense causal attention. Otherwise
    ``max(1, ceil(sparse_attn_prev_fraction * num_sites))``.
    """
    f = float(sparse_attn_prev_fraction)
    if f <= 0.0:
        return None
    if f > 1.0:
        raise ValueError(f"sparse_attn_prev_fraction must be in [0, 1], got {f}")
    return max(1, int(math.ceil(f * float(num_sites))))


def _attention_additive_mask(
    seq_len: int,
    device: torch.device,
    attn_window_tokens: Optional[int],
) -> torch.Tensor:
    """
    Additive attention mask ``(L, L)``: ``0`` allowed, ``-inf`` blocked.

    Full causal attention when ``attn_window_tokens is None``. Otherwise causal **band** attention:
    query row ``i`` may attend only to key columns ``j`` with ``max(0, i - W + 1) <= j <= i``.
    """
    if attn_window_tokens is None:
        return _causal_attention_mask(seq_len, device)
    w = int(attn_window_tokens)
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    causal_ok = j <= i
    low = (i - w + 1).clamp(min=0)
    band_ok = j >= low
    ok = causal_ok & band_ok
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=torch.float32)
    zero = torch.zeros((), device=device, dtype=torch.float32)
    return torch.where(ok, zero, neg_inf)


def _apply_topk_causal_to_scores(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """
    Keep only the largest ``topk`` attention logits per query row among **causal** keys (columns ``≤`` row).

    Parameters
    ----------
    scores : (B, H, L, L)
        Pre-softmax scores; upper triangle past diagonal should already be ``-inf`` (causal).
    topk : int
        Must be ``>= 1``. Rows with fewer than ``topk`` valid keys are unchanged.
    """
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")
    out = scores.clone()
    _, _, L, _ = out.shape
    for i in range(L):
        width = i + 1
        tk = min(topk, width)
        if tk >= width:
            continue
        row = out[:, :, i, :width]
        _, idx = torch.topk(row, tk, dim=-1)
        keep = torch.zeros_like(row, dtype=torch.bool)
        keep.scatter_(-1, idx, True)
        out[:, :, i, :width] = row.masked_fill(~keep, float("-inf"))
    return out


def _transformer_encoder_layer_forward_seq_topk(
    layer: nn.TransformerEncoderLayer,
    x: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Full-sequence forward with causal **top-k** sparse attention (post-norm layer)."""
    if layer.norm_first:
        raise NotImplementedError("top-k path requires norm_first=False.")
    attn = layer.self_attn
    q, k, v = _mha_in_proj_qkv(attn, x)
    dh = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(dh))
    L = scores.shape[-1]
    causal = _causal_attention_mask(L, scores.device).to(dtype=scores.dtype)
    scores = scores + causal.unsqueeze(0).unsqueeze(0)
    scores = _apply_topk_causal_to_scores(scores, topk)
    p = F.softmax(scores, dim=-1)
    attn_vec = torch.matmul(p, v)
    bsz, h, ln, _hd = attn_vec.shape
    attn_vec = attn_vec.transpose(1, 2).contiguous().reshape(bsz, ln, attn.embed_dim)
    attn_vec = attn.out_proj(attn_vec)
    attn_vec = layer.dropout1(attn_vec)
    x_mid = layer.norm1(x + attn_vec)
    ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_mid))))
    ff = layer.dropout2(ff)
    return layer.norm2(x_mid + ff)


def _mha_in_proj_qkv(
    attn: nn.MultiheadAttention,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Same Q/K/V split as ``nn.MultiheadAttention`` for batch-first input ``x`` (B, T, E).

    Returns
    -------
    q, k, v : each (B, H, T, head_dim)
    """
    if x.dim() != 3:
        raise ValueError(f"expected (B, T, E), got {tuple(x.shape)}")
    bsz, tgt_len, embed_dim = x.shape
    if embed_dim != attn.embed_dim:
        raise ValueError("embed_dim mismatch")
    h = attn.num_heads
    head_dim = embed_dim // h
    proj = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
    proj = proj.unflatten(-1, (3, embed_dim))
    q, k, v = proj.unbind(dim=2)
    q = q.reshape(bsz, tgt_len, h, head_dim).transpose(1, 2)
    k = k.reshape(bsz, tgt_len, h, head_dim).transpose(1, 2)
    v = v.reshape(bsz, tgt_len, h, head_dim).transpose(1, 2)
    return q, k, v


def _mha_sliding_causal_band_attention(
    attn: nn.MultiheadAttention,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """
    Causal sliding-window self-attention without materializing an ``L×L`` score matrix.

    Query row ``i`` only attends to keys ``max(0, i - Wm + 1) … i`` with
    ``Wm = min(window, L)``, matching :func:`_attention_additive_mask` for the same ``window``.

    Parameters
    ----------
    q, k, v : (B, H, L, Dh)
    window : int
        Global band size ``W`` (from ``ceil(fraction · num_sites)``).

    Returns
    -------
    Tensor (B, L, embed_dim), i.e. after ``out_proj`` (same layout as ``nn.MultiheadAttention`` batch_first).
    """
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, v must be (B, H, L, Dh)")
    B, H, L, Dh = q.shape
    Wm = min(int(window), int(L))
    device = q.device
    scale = 1.0 / math.sqrt(float(Dh))
    w_idx = torch.arange(Wm, device=device, dtype=torch.long).view(1, Wm)
    i_idx = torch.arange(L, device=device, dtype=torch.long).view(L, 1)
    key_idx = i_idx - (Wm - 1) + w_idx
    valid = key_idx >= 0
    key_idx_s = key_idx.clamp(0, L - 1).long()
    k_g = k[:, :, key_idx_s, :]
    v_g = v[:, :, key_idx_s, :]
    scores = (q.unsqueeze(3) * k_g).sum(dim=-1) * scale
    scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float("-inf"))
    p = F.softmax(scores, dim=-1)
    ctx = (p.unsqueeze(-1) * v_g).sum(dim=3)
    ctx = ctx.transpose(1, 2).contiguous().reshape(B, L, H * Dh)
    return attn.out_proj(ctx)


def _layer_step_cached(
    layer: nn.TransformerEncoderLayer,
    h_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    pattern: str,
    cap: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One post-norm encoder layer on a **single new token**, attending into cached K/V.

    ``k_cache`` / ``v_cache`` are ``(B, H, T, Dh)`` for positions already decoded (``T`` may be 0).
    Returns the new hidden ``(B, 1, E)`` and the K/V sequence **including** this token.

    ``pattern`` is ``"dense"`` (all past keys), ``"band"`` (last ``cap`` keys), or ``"topk"``
    (top ``cap`` keys among all past keys, same rule as :func:`_apply_topk_causal_to_scores`
    on the new query only).
    """
    if layer.norm_first:
        raise NotImplementedError("cached step requires norm_first=False.")
    if h_new.shape[1] != 1:
        raise ValueError(f"h_new must be (B, 1, E), got {tuple(h_new.shape)}")
    attn = layer.self_attn
    q, k, v = _mha_in_proj_qkv(attn, h_new)
    if k_cache.shape[2] == 0:
        k_all, v_all = k, v
    else:
        k_all = torch.cat([k_cache, k], dim=2)
        v_all = torch.cat([v_cache, v], dim=2)
    length = int(k_all.shape[2])
    if pattern == "dense":
        k_w, v_w = k_all, v_all
        scores = torch.matmul(q, k_w.transpose(-2, -1)) / math.sqrt(float(q.shape[-1]))
    elif pattern == "band":
        if cap is None:
            raise ValueError("band cache step requires cap")
        wm = min(int(cap), length)
        k_w = k_all[:, :, -wm:, :]
        v_w = v_all[:, :, -wm:, :]
        scores = torch.matmul(q, k_w.transpose(-2, -1)) / math.sqrt(float(q.shape[-1]))
    elif pattern == "topk":
        if cap is None:
            raise ValueError("topk cache step requires cap")
        k_w, v_w = k_all, v_all
        scores = torch.matmul(q, k_w.transpose(-2, -1)) / math.sqrt(float(q.shape[-1]))
        tk = min(int(cap), length)
        if tk < length:
            _, idx = torch.topk(scores, tk, dim=-1)
            keep = torch.zeros_like(scores, dtype=torch.bool)
            keep.scatter_(-1, idx, True)
            scores = scores.masked_fill(~keep, float("-inf"))
    else:
        raise ValueError(f"unknown cache attention pattern {pattern!r}")
    p = F.softmax(scores, dim=-1)
    ctx = torch.matmul(p, v_w)
    bsz, heads, _, _hd = ctx.shape
    ctx = ctx.transpose(1, 2).contiguous().reshape(bsz, 1, heads * _hd)
    attn_vec = layer.dropout1(attn.out_proj(ctx))
    x_mid = layer.norm1(h_new + attn_vec)
    ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_mid))))
    ff = layer.dropout2(ff)
    return layer.norm2(x_mid + ff), k_all, v_all


def _transformer_encoder_layer_forward_seq_sliding(
    layer: nn.TransformerEncoderLayer,
    x: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """One encoder layer, post-norm, with sliding-window attention (``band`` training path)."""
    if layer.norm_first:
        raise NotImplementedError("sliding band path requires norm_first=False.")
    attn = layer.self_attn
    q, k, v = _mha_in_proj_qkv(attn, x)
    attn_vec = _mha_sliding_causal_band_attention(attn, q, k, v, window)
    attn_vec = layer.dropout1(attn_vec)
    x_mid = layer.norm1(x + attn_vec)
    ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_mid))))
    ff = layer.dropout2(ff)
    return layer.norm2(x_mid + ff)


class AutoregressiveBitModel(nn.Module):
    """
    Causal Transformer LM over ``N`` binary variables (vocabulary ``{0, 1}``, plus BOS).

    Forward pass predicts a distribution over each ``x_k`` given ``x_{<k}``.

    Optional sparse attention when ``sparse_attn_prev_fraction > 0`` (budget
    ``K = ceil(fraction · N)`` keys per query):

    * ``sparse_attn_pattern='band'`` — **Sliding last-K window**: each query only attends to the last
      ``K`` keys. Training uses a true sliding implementation (``O(L·K)`` per layer, no full ``L×L``
      score matrix).

    * ``sparse_attn_pattern='topk'`` — **learned** sparse attention: full causal scores ``QKᵀ/√d``,
      then keep only the top ``K`` logits per query row before softmax. Which positions survive is
      context-dependent (via learned projections). Training still builds an ``L×L`` score matrix
      per layer (same asymptotics as dense attention).
    """

    BOS_ID = 2

    def __init__(
        self,
        num_sites: int,
        d_model: int = 128,
        nhead: int = 4,
        nlayers: int = 4,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.0,
        sparse_attn_prev_fraction: float = 0.0,
        sparse_attn_pattern: str = "band",
    ):
        super().__init__()
        self.num_sites = int(num_sites)
        self.d_model = int(d_model)
        sf = float(sparse_attn_prev_fraction)
        if sf < 0.0 or sf > 1.0:
            raise ValueError(
                f"sparse_attn_prev_fraction must be in [0, 1], got {sparse_attn_prev_fraction}"
            )
        self.sparse_attn_prev_fraction = sf
        pat = str(sparse_attn_pattern).strip().lower()
        if pat not in ("band", "topk"):
            raise ValueError(
                f"sparse_attn_pattern must be 'band' or 'topk', got {sparse_attn_pattern!r}"
            )
        self.sparse_attn_pattern = pat
        self._sparse_attn_window: Optional[int] = sparse_attn_window_tokens(
            self.num_sites, sf
        )
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        df = int(dim_feedforward) if dim_feedforward is not None else 4 * d_model
        self.tok_emb = nn.Embedding(3, d_model)
        self.pos_emb = nn.Embedding(num_sites, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=df,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.lm_head = nn.Linear(d_model, 2)

    def _forward_encoder_full_sequence_topk(self, h: torch.Tensor) -> torch.Tensor:
        """Run all encoder layers with causal top-k sparse attention."""
        cap = self._sparse_attn_window
        if cap is None:
            raise RuntimeError("top-k encoder requires sparse_attn_prev_fraction > 0.")
        topk = int(cap)
        for layer in self.encoder.layers:
            h = _transformer_encoder_layer_forward_seq_topk(layer, h, topk)
        return h

    def _forward_encoder_full_sequence_band_sliding(self, h: torch.Tensor) -> torch.Tensor:
        """Run all encoder layers with causal **sliding-window** attention (no full ``L×L`` scores)."""
        cap = self._sparse_attn_window
        if cap is None:
            raise RuntimeError("band sliding requires sparse_attn_prev_fraction > 0.")
        W = int(cap)
        for layer in self.encoder.layers:
            h = _transformer_encoder_layer_forward_seq_sliding(layer, h, W)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, N) int64 with values 0 or 1.

        Returns
        -------
        logits : (B, N, 2) for bits ``x_0 … x_{N-1}`` (position ``k`` uses prefix ``x_{<k}`` and BOS).
        """
        b, n = x.shape
        if n != self.num_sites:
            raise ValueError(f"x has length {n}, expected num_sites={self.num_sites}")
        device = x.device
        bos = torch.full((b, 1), self.BOS_ID, dtype=torch.long, device=device)
        inp_ids = torch.cat([bos, x[:, :-1]], dim=1)
        tok = self.tok_emb(inp_ids)
        pos_ids = torch.arange(n, device=device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        h = tok + self.pos_emb(pos_ids)
        if self._sparse_attn_window is None:
            attn_mask = _causal_attention_mask(n, device)
            h = self.encoder(h, mask=attn_mask)
        elif self.sparse_attn_pattern == "topk":
            h = self._forward_encoder_full_sequence_topk(h)
        else:
            h = self._forward_encoder_full_sequence_band_sliding(h)
        return self.lm_head(h)

    def forward_next(self, x_prefix: torch.Tensor) -> torch.Tensor:
        """
        Next-bit logits from autoregressive context ``x_prefix`` (bits ``x_0 … x_{t-1}``).

        Parameters
        ----------
        x_prefix : (B, t) int64 with ``t`` in ``[0, num_sites]``; ``t == 0`` means **no** prior bit was
            chosen yet — predict ``x_0`` from **BOS only** (not a literal ``x_{-1} = 0`` token).

        Returns
        -------
        (B, 2) logits for the bit at position ``t``.
        """
        b, t = x_prefix.shape
        device = x_prefix.device
        if t == 0:
            inp_ids = torch.full((b, 1), self.BOS_ID, dtype=torch.long, device=device)
        else:
            bos = torch.full((b, 1), self.BOS_ID, dtype=torch.long, device=device)
            inp_ids = torch.cat([bos, x_prefix], dim=1)
        seq_len = inp_ids.shape[1]
        if seq_len > self.num_sites:
            raise ValueError("Prefix too long for num_sites.")
        tok = self.tok_emb(inp_ids)
        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        h = tok + self.pos_emb(pos_ids)
        if self._sparse_attn_window is None:
            attn_mask = _causal_attention_mask(seq_len, device)
            h = self.encoder(h, mask=attn_mask)
        elif self.sparse_attn_pattern == "topk":
            h = self._forward_encoder_full_sequence_topk(h)
        else:
            h = self._forward_encoder_full_sequence_band_sliding(h)
        return self.lm_head(h[:, -1, :])

    def forward_next_soft(self, prefix_prob1: torch.Tensor) -> torch.Tensor:
        """
        Next-bit logits with a **soft** binary prefix: each past site is the probability of bit ``1``.

        Token embeddings mix ``E(0)`` and ``E(1)`` with weights ``(1-p)`` and ``p``; BOS is unchanged.
        Use this path for differentiable objectives (e.g. relaxed Lagrangian over ``Ax``).

        Parameters
        ----------
        prefix_prob1 : (B, t) float, values in ``[0, 1]`` — ``P(x_k = 1)`` for ``k = 0 … t-1``.
            ``t == 0`` means empty prefix (BOS-only), same as :meth:`forward_next` with length-0 prefix.

        Returns
        -------
        (B, 2) logits for the bit at site ``t``.
        """
        if prefix_prob1.ndim != 2:
            raise ValueError(f"prefix_prob1 must be (B, t), got shape {prefix_prob1.shape}")
        b, t = prefix_prob1.shape
        device = prefix_prob1.device
        w = self.tok_emb.weight
        if t == 0:
            tok = w[self.BOS_ID : self.BOS_ID + 1].to(device=device, dtype=torch.float32).expand(b, 1, -1)
        else:
            p = prefix_prob1.clamp(0.0, 1.0).to(dtype=torch.float32)
            emb0 = w[0].to(device=device, dtype=torch.float32).view(1, 1, -1)
            emb1 = w[1].to(device=device, dtype=torch.float32).view(1, 1, -1)
            emb_bit = (1.0 - p).unsqueeze(-1) * emb0 + p.unsqueeze(-1) * emb1
            bos = w[self.BOS_ID : self.BOS_ID + 1].to(device=device, dtype=torch.float32).expand(b, 1, -1)
            tok = torch.cat([bos, emb_bit], dim=1)
        seq_len = tok.shape[1]
        if seq_len > self.num_sites:
            raise ValueError("Prefix too long for num_sites.")
        pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        h = tok + self.pos_emb(pos_ids).float()
        if self._sparse_attn_window is None:
            attn_mask = _causal_attention_mask(seq_len, device)
            h = self.encoder(h, mask=attn_mask)
        elif self.sparse_attn_pattern == "topk":
            h = self._forward_encoder_full_sequence_topk(h)
        else:
            h = self._forward_encoder_full_sequence_band_sliding(h)
        return self.lm_head(h[:, -1, :])

    def init_soft_kv_cache(
        self, batch: int, device: torch.device
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Empty per-layer K/V cache for :meth:`forward_next_soft_cached`."""
        attn0 = self.encoder.layers[0].self_attn
        head_dim = int(attn0.embed_dim // attn0.num_heads)
        nhead = int(attn0.num_heads)
        caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in self.encoder.layers:
            k = torch.zeros(int(batch), nhead, 0, head_dim, device=device, dtype=torch.float32)
            v = torch.zeros(int(batch), nhead, 0, head_dim, device=device, dtype=torch.float32)
            caches.append((k, v))
        return caches

    def _embed_new_soft_token(self, prefix_prob1: torch.Tensor) -> torch.Tensor:
        """Embedding of the next encoder token given a soft prefix of length ``t`` (``(B, 1, E)``)."""
        if prefix_prob1.ndim != 2:
            raise ValueError(f"prefix_prob1 must be (B, t), got shape {prefix_prob1.shape}")
        b, t = prefix_prob1.shape
        device = prefix_prob1.device
        w = self.tok_emb.weight
        if t == 0:
            tok = w[self.BOS_ID : self.BOS_ID + 1].to(device=device, dtype=torch.float32).expand(b, 1, -1)
        else:
            p = prefix_prob1[:, t - 1 : t].clamp(0.0, 1.0).to(dtype=torch.float32)
            emb0 = w[0].to(device=device, dtype=torch.float32)
            emb1 = w[1].to(device=device, dtype=torch.float32)
            tok = ((1.0 - p) * emb0 + p * emb1).unsqueeze(1)
        pos = self.pos_emb.weight[t].to(device=device, dtype=torch.float32)
        return tok + pos.view(1, 1, -1)

    def forward_next_soft_cached(
        self,
        prefix_prob1: torch.Tensor,
        caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Same logits as :meth:`forward_next_soft`, but only the new token is encoded.

        Past keys/values are read from ``caches`` (one ``(K, V)`` pair per encoder layer,
        ``K,V`` shaped ``(B, H, T, Dh)`` with ``T == prefix_prob1.shape[1]``). Causal band /
        dense / top-k attention on that new query matches the full-prefix forward when dropout is 0.
        """
        if len(caches) != len(self.encoder.layers):
            raise ValueError(
                f"expected {len(self.encoder.layers)} cache layers, got {len(caches)}"
            )
        t = int(prefix_prob1.shape[1])
        if int(caches[0][0].shape[2]) != t:
            raise ValueError(
                f"cache length {int(caches[0][0].shape[2])} != prefix length {t}"
            )
        h = self._embed_new_soft_token(prefix_prob1)
        if self._sparse_attn_window is None:
            pattern = "dense"
            cap = None
        else:
            pattern = self.sparse_attn_pattern
            cap = int(self._sparse_attn_window)
        new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer, (k_c, v_c) in zip(self.encoder.layers, caches):
            h, k_all, v_all = _layer_step_cached(
                layer, h, k_c, v_c, pattern=pattern, cap=cap
            )
            new_caches.append((k_all, v_all))
        return self.lm_head(h[:, 0, :]), new_caches
