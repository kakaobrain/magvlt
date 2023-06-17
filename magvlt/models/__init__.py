from magvlt.models.stage1.vqgan import VQGAN
from magvlt.models.stage2.transformers import Transformer1d
from magvlt.models.maskgit.maskgit import MaskGITModel
from magvlt.models.utils import SPECIAL_TOKENS_BASIC, SPECIAL_TOKENS_HNH


def build_model(cfg, tokenizer=None):
    # stage 1
    if cfg.stage1.type == "vqgan":  # TODO: change type -> backbone
        model_vq = VQGAN(
            n_embed=cfg.stage1.n_embed,
            embed_dim=cfg.stage1.embed_dim,
            hparams=cfg.stage1.hparams,
        )
    else:
        raise ValueError(f"{cfg.stage1.type} is invalid backbone...")

    model_vq.eval()  # may not necessary
    for p in model_vq.parameters():
        p.requires_grad = False

    spc_tok_class = SPECIAL_TOKENS_BASIC
    use_hnh_task = getattr(cfg.dataset, "use_hnh_task", False)
    if use_hnh_task:
        spc_tok_class = SPECIAL_TOKENS_HNH

    # stage 2
    if cfg.stage2.backbone == "transformer1d":
        model = Transformer1d(
            vocab_size_txt=cfg.stage2.vocab_size_txt,
            vocab_size_img=cfg.stage2.vocab_size_img,
            hparams=cfg.stage2.hparams,
            spc_tok_class=spc_tok_class,
        )
    else:
        raise ValueError(f"{cfg.stage2.backbone} is invalid backbone...")

    if cfg.stage2.type == "maskgit":
        return MaskGITModel(cfg, model, model_vq, tokenizer)
    else:
        raise ValueError(f"{cfg.stage2.type} is invalid type...")
