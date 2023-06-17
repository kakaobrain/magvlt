import torch
import torch.nn as nn
import math


class PositionalEmbed(nn.Module):
    def __init__(self, mode="la1d", length=64, embed_dim=1024):
        super().__init__()
        self.mode = mode
        self.length = length
        if self.mode == "la1d":
            self.pos_embed = nn.Parameter(torch.zeros(length, embed_dim))
            self.pos_embed.data.normal_(mean=0.0, std=0.02)
        elif self.mode == "s2d":
            self.pos_embed = Sinusoidal2DPositionalEmbed(
                length=length, embed_dim=embed_dim, learnable=False
            )
        elif self.mode == "ls2d":
            self.pos_embed = Sinusoidal2DPositionalEmbed(
                length=length, embed_dim=embed_dim, learnable=True
            )
        elif self.mode == "s1d":
            self.pos_embed = Sinusoidal1DPositionalEmbed(
                length=length, embed_dim=embed_dim, learnable=False
            )
        elif self.mode == "ls1d":
            self.pos_embed = Sinusoidal1DPositionalEmbed(
                length=length, embed_dim=embed_dim, learnable=True
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if self.mode == "la1d":
            pos_embed = self.pos_embed[x, :]
        else:
            pos_embed = self.pos_embed(x)
        return pos_embed


class Sinusoidal1DPositionalEmbed(nn.Module):
    def __init__(self, length=64, embed_dim=1024, learnable=True):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dimension (got dim={:d})".format(embed_dim)
            )
        self.length = length
        self.embed_dim = embed_dim
        self.dim = self.embed_dim

        div = torch.exp(
            torch.arange(0.0, self.dim, 2) * -(math.log(10000.0) / self.dim)
        )
        if learnable:
            self.div = nn.Parameter(div)
        else:
            self.register_buffer("div", div)

    def forward(self, x):
        B = x.shape[0]
        L = x.shape[1]

        pos_embed = torch.zeros(B, L, self.embed_dim).to(x.device)
        pos_embed[:, :, 0 : self.dim : 2] += torch.sin(x.unsqueeze(-1) * self.div)
        pos_embed[:, :, 1 : self.dim : 2] += torch.cos(x.unsqueeze(-1) * self.div)

        return pos_embed


class Sinusoidal2DPositionalEmbed(nn.Module):
    def __init__(self, length=256, embed_dim=1024, learnable=True):
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dimension (got dim={:d})".format(embed_dim)
            )
        self.length = length
        self.HW = int(self.length**0.5)
        self.embed_dim = embed_dim
        self.dim = int(self.embed_dim / 2)

        div = torch.exp(
            torch.arange(0.0, self.dim, 2) * -(math.log(10000.0) / self.dim)
        )
        div = div.repeat([2])
        if learnable:
            self.div = nn.Parameter(div)
        else:
            self.register_buffer("div", div)

        self.hw_idx = torch.tensor(
            [[[h, w] for w in range(self.HW)] for h in range(self.HW)]
        ).to(torch.float)
        self.hw_idx = self.hw_idx.reshape([self.length, 2])

    def forward(self, x):
        B = x.shape[0]
        L = x.shape[1]

        pos_w = self.hw_idx[x, 1].unsqueeze(-1).to(x.device)
        pos_h = self.hw_idx[x, 0].unsqueeze(-1).to(x.device)
        pos_embed = torch.zeros(B, L, self.embed_dim).to(x.device)

        pos_embed[:, :, 0 : self.dim : 2] += torch.sin(
            pos_w * self.div[: self.dim // 2]
        ).reshape([B, L, -1])
        pos_embed[:, :, 1 : self.dim : 2] += torch.cos(
            pos_w * self.div[: self.dim // 2]
        ).reshape([B, L, -1])
        pos_embed[:, :, self.dim :: 2] += torch.sin(
            pos_h * self.div[self.dim // 2 :]
        ).reshape([B, L, -1])
        pos_embed[:, :, self.dim + 1 :: 2] += torch.cos(
            pos_h * self.div[self.dim // 2 :]
        ).reshape([B, L, -1])

        return pos_embed
