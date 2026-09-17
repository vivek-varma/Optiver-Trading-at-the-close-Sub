"""Optimized (stateful / incremental) inference for the sequence models.

The naive submission loop re-runs each net on the full sequence [0..t] at every
bucket just to read the output at step t. Because both nets are strictly causal,
the outputs for steps 0..t-1 are identical to what was computed at the previous
bucket -- so we carry internal state forward and process only the new step.

  GRU        : carry the hidden state h. Exact. O(T) -> O(1) per bucket.
  Transformer: cache each layer's keys/values (a "KV cache", the same trick LLM
               inference uses). Exact. O(T^2) -> O(T) per bucket.
"""
import math
import torch
import torch.nn.functional as Fn


class StatefulGRU:
    """Wraps a trained SeqModel; carries the GRU hidden state across buckets."""

    def __init__(self, model):
        self.m = model
        self.h = None                       # (layers, n_stocks, hidden)

    def reset(self):
        self.h = None

    @torch.no_grad()
    def step(self, x_t, sidx):
        """x_t: (n_stocks, n_feat) scaled features for the CURRENT step only.
        sidx: (n_stocks,) stock indices, in a FIXED order for the whole day."""
        e = self.m.emb(sidx)                                   # (n, emb)
        inp = torch.cat([x_t, e], -1).unsqueeze(1)             # (n, 1, nf+emb)
        out, self.h = self.m.rnn(inp, self.h)                  # h carried forward
        return self.m.head(out[:, -1]).squeeze(-1)             # (n,)


class KVCacheTransformer:
    """Incremental inference for the causal TimeTransformer with a per-layer
    KV cache. Reproduces nn.TransformerEncoderLayer (post-norm, batch_first,
    GELU) exactly, but computes only the new position each step."""

    def __init__(self, model):
        self.m = model
        self.layers = list(model.encoder.layers)
        self.d = model.in_proj.out_features
        self.nhead = self.layers[0].self_attn.num_heads
        self.hd = self.d // self.nhead
        self.reset()

    def reset(self):
        self.t = 0
        self.K = [None] * len(self.layers)  # each: (n, heads, T, hd)
        self.V = [None] * len(self.layers)

    @torch.no_grad()
    def step(self, x_t, sidx):
        m = self.m
        e = m.emb(sidx)
        h = m.in_proj(torch.cat([x_t, e], -1)) + m.pos[0, self.t]       # (n, d)
        n = h.size(0)
        for li, L in enumerate(self.layers):
            at = L.self_attn
            # q, k, v for the NEW position only
            qkv = Fn.linear(h, at.in_proj_weight, at.in_proj_bias)      # (n, 3d)
            q, k, v = qkv.split(self.d, dim=-1)
            q = q.view(n, self.nhead, 1, self.hd)
            k = k.view(n, self.nhead, 1, self.hd)
            v = v.view(n, self.nhead, 1, self.hd)
            # append to cache
            self.K[li] = k if self.K[li] is None else torch.cat([self.K[li], k], dim=2)
            self.V[li] = v if self.V[li] is None else torch.cat([self.V[li], v], dim=2)
            # attend the single new query over all cached keys (causal by construction)
            scores = (q @ self.K[li].transpose(-1, -2)) / math.sqrt(self.hd)  # (n, heads, 1, T)
            attn = torch.softmax(scores, dim=-1) @ self.V[li]                   # (n, heads, 1, hd)
            sa = at.out_proj(attn.reshape(n, self.d))                           # (n, d)
            # post-norm residual blocks, identical to nn.TransformerEncoderLayer
            h = L.norm1(h + sa)
            ff = L.linear2(Fn.gelu(L.linear1(h)))
            h = L.norm2(h + ff)
        self.t += 1
        return m.head(h).squeeze(-1)                                            # (n,)
