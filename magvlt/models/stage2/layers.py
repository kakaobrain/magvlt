import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class GELU(nn.Module):
    def __init__(self, use_approx=False):
        super().__init__()
        self.use_approx = use_approx

    def forward(self, x):
        if self.use_approx:
            return x * torch.sigmoid(1.702 * x)
        else:
            return F.gelu(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        ctx_len: int,
        embed_dim: int,
        n_heads: int,
        resid_pdrop: float,
        attn_pdrop: float,
        attn_bias: bool,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0

        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.query = nn.Linear(embed_dim, embed_dim, bias=attn_bias)
        self.value = nn.Linear(embed_dim, embed_dim, bias=attn_bias)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # output projection
        self.proj = nn.Linear(embed_dim, embed_dim, attn_bias)

        self.n_heads = n_heads
        self.ctx_len = ctx_len

    def forward(
        self,
        x,
        use_cache=False,
        layer_past=None,
        attn_mask=None,
    ):
        B, T, C = x.shape  # B=4,  T=320, C=1024
        x = x.transpose(0, 1).contiguous()  # (B, T, C) -> (T, B, C)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(T, B * self.n_heads, C // self.n_heads).transpose(0, 1)
        )  # (B*nh, T, hs)
        q = (
            self.query(x).view(T, B * self.n_heads, C // self.n_heads).transpose(0, 1)
        )  # (B*nh, T, hs)
        v = (
            self.value(x).view(T, B * self.n_heads, C // self.n_heads).transpose(0, 1)
        )  # (B*nh, T, hs)

        if use_cache:
            present = torch.stack([k, v])

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)

        # Tensor shape below: (B * nh, T, hs) X (B * nh, hs, T) -> (B * nh, T, T)
        att = torch.bmm(q, (k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))
        if attn_mask is not None:
            no_attn_ids = torch.logical_not(attn_mask.sum(dim=-1))
            attn_mask[torch.diag_embed(no_attn_ids)] = True  # at least one attention
            mask = torch.repeat_interleave(attn_mask, self.n_heads, dim=0)
            att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = torch.bmm(
            att, v
        )  # (B*nh, T, T) X (B*nh, T, hs) -> (B*nh, T, hs) 32 x 320 x 128
        y = (
            y.transpose(0, 1).contiguous().view(T, B, C)
        )  # re-assemble all head outputs side by side 320 x 4 x 1024

        # output projection
        y = self.resid_drop(self.proj(y))  # 320 x 4 x 1024
        if use_cache:
            return y.transpose(0, 1).contiguous(), present  # (T, B, C) -> (B, T, C)
        else:
            return y.transpose(0, 1).contiguous()  # (T, B, C) -> (B, T, C)

    def sample(self, x, layer_past=None):
        x, present = self(x, use_cache=True, layer_past=layer_past)
        return x, present


class Block(nn.Module):
    def __init__(
        self,
        ctx_len: int,
        embed_dim: int,
        n_heads: int,
        mlp_bias: bool,
        attn_bias: bool,
        resid_pdrop: bool,
        attn_pdrop: bool,
        gelu_use_approx: bool,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadSelfAttention(
            ctx_len=ctx_len,
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            attn_bias=attn_bias,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=mlp_bias),
            GELU(gelu_use_approx),
            nn.Linear(4 * embed_dim, embed_dim, bias=mlp_bias),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(
            self.ln1(x),
            attn_mask=attn_mask,
        )
        x = x + self.mlp(self.ln2(x))
        return x  # 4 x 320 x 1024

    def sample(self, x, layer_past=None):
        attn, present = self.attn.sample(self.ln1(x), layer_past=layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x, present
