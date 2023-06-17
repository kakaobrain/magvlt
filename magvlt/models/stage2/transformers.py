import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from omegaconf import OmegaConf
from magvlt.models.stage2.position_emb import PositionalEmbed
from magvlt.models.stage2.layers import Block, GELU
from magvlt.models.utils import SPECIAL_TOKENS


class ModelTemplate(nn.Module):
    def __init__(self):
        super().__init__()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size


class LMLayer(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, embedding):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, intermediate_dim)
        self.gelu = GELU()
        self.ln = nn.LayerNorm(intermediate_dim)
        self.output_bias = torch.nn.Parameter(torch.zeros(embedding.weight.shape[0]))

    def forward(self, x, embedding):
        h = self.ln(self.gelu(self.linear(x)))
        logits = torch.matmul(h, embedding.weight.T) + self.output_bias
        return logits


class Transformer1d(ModelTemplate):
    def __init__(
        self,
        vocab_size_txt: int,
        vocab_size_img: int,
        hparams: OmegaConf,
        spc_tok_class: SPECIAL_TOKENS,
    ):
        super().__init__()

        self.spc_tok_class = spc_tok_class

        # input embedding for image and text
        self.vocab_size_img = vocab_size_img
        self.vocab_size_txt = vocab_size_txt

        self.tok_emb = nn.Embedding(vocab_size_img + vocab_size_txt, hparams.embed_dim)
        self.tok_emb_spc = nn.Embedding(spc_tok_class.get_length(), hparams.embed_dim)

        self.drop = nn.Dropout(hparams.embd_pdrop)

        # transformer blocks
        self.blocks = [
            Block(
                ctx_len=hparams.ctx_len_img + hparams.ctx_len_txt,
                embed_dim=hparams.embed_dim,
                n_heads=hparams.n_heads,
                mlp_bias=hparams.mlp_bias,
                attn_bias=hparams.attn_bias,
                resid_pdrop=hparams.resid_pdrop,
                attn_pdrop=hparams.attn_pdrop,
                gelu_use_approx=hparams.gelu_use_approx,
            )
            for i in range(1, hparams.n_layers + 1)
        ]
        self.blocks = nn.Sequential(*self.blocks)

        # heads for image and text
        use_lm_layer = getattr(hparams, "use_lm_layer", True)
        if use_lm_layer:
            self.lm_layer = LMLayer(hparams.embed_dim, hparams.embed_dim, self.tok_emb)

        self.ctx_len_img = hparams.ctx_len_img
        self.ctx_len_txt = hparams.ctx_len_txt
        self.n_layers = hparams.n_layers
        self.use_target_vocab_only = hparams.use_target_vocab_only
        self.use_spc_pos = getattr(hparams, "use_spc_pos", False)

        self.pos_emb_img = None
        self.pos_emb_txt = None
        self.pos_emb_spc = None

        self.apply(self._init_weights)

        # PositionalEmbed modules should be located here. Weights initialization is done in inside the module.
        self.pos_emb_img = PositionalEmbed(
            mode=hparams.pos_emb_img_mode,
            length=hparams.ctx_len_img,
            embed_dim=hparams.embed_dim,
        )
        self.pos_emb_txt = PositionalEmbed(
            mode=hparams.pos_emb_txt_mode,
            length=hparams.ctx_len_txt,
            embed_dim=hparams.embed_dim,
        )
        self.pos_emb_spc = PositionalEmbed(
            mode=hparams.pos_emb_spc_mode,
            length=hparams.ctx_len_txt + hparams.ctx_len_img + 4,
            embed_dim=hparams.embed_dim,
        )

    def tok_emb_txt(self, texts):
        return self.tok_emb(self.vocab_size_img + texts)

    def forward(self, x, pos, loc_img, loc_txt, loc_spc, attn_mask=None, amp=True):
        B = x.shape[0]
        L = x.shape[1]
        loc_img_nonmask = torch.logical_and(loc_img, torch.logical_not(loc_spc))
        loc_txt_nonmask = torch.logical_and(loc_txt, torch.logical_not(loc_spc))

        tok_img = x[loc_img_nonmask]
        tok_txt = x[loc_txt_nonmask]
        tok_spc = x[loc_spc]

        with autocast(enabled=amp):

            emb_txt = self.tok_emb_txt(tok_txt)
            # pos_emb_txt = self.pos_emb_txt(pos[loc_txt]).squeeze(0)

            emb_img = self.tok_emb(tok_img)
            # pos_emb_img = self.pos_emb_img(pos[loc_img]).squeeze(0)

            emb_spc = self.tok_emb_spc(tok_spc)

            x = torch.zeros([B, L, emb_img.shape[-1]]).to(x.device)
            x[loc_img_nonmask] += emb_img
            x[loc_txt_nonmask] += emb_txt
            x[loc_spc] += emb_spc

            if self.pos_emb_img is not None:
                pos_emb_img = self.pos_emb_img(pos[loc_img]).squeeze(0)
                x[loc_img] += pos_emb_img
            if self.pos_emb_txt is not None:
                pos_emb_txt = self.pos_emb_txt(pos[loc_txt]).squeeze(0)
                x[loc_txt] += pos_emb_txt
            if self.use_spc_pos and self.pos_emb_spc is not None:
                pos_emb_spc = self.pos_emb_spc(pos[loc_spc]).squeeze(0)
                x[loc_spc] += pos_emb_spc

            x = self.drop(x)
            for block in self.blocks:
                x = block(
                    x,
                    attn_mask=attn_mask,
                )  # 4 x 320 x 1024
        return x
