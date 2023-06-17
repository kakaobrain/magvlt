import argparse
import time

import clip
import torchvision.transforms as transforms
import pytorch_lightning as pl
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from magvlt.datamodules.tokenizers import TokenizerUtils
from magvlt.datamodules.dataclasses import TextInputItem
from magvlt.models import build_model
from magvlt.models.utils import clip_score


def default_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_img_path",
        type=str,
        default="assets/coco_sample.png",
        help="A path to source image",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to a model checkpoint"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to a model config"
    )
    parser.add_argument(
        "--stage1_model_path",
        type=str,
        required=True,
        help="Path to a stage1 model checkpoint",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed")

    return parser


if __name__ == "__main__":
    parser = default_parser()
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_path)
    cfg.stage2.mask_hparams.i2t_n_steps = cfg.sampling.txt_num_steps

    pl.seed_everything(args.seed)

    tokutil = TokenizerUtils()
    tokutil.build_tokenizer(
        cfg.dataset.tokenizer.type,
        cfg.dataset.tokenizer.hparams.context_length,
        lowercase=True,
        dropout=None,
    )

    model_pl = build_model(cfg, tokenizer=tokutil.tokenizer)
    model_pl.load_model(ckpt_path=args.model_path)
    model_pl.load_stage1_model(ckpt_path=args.stage1_model_path)
    model_pl.eval()
    model_pl.cuda()

    model_clip, preprocess_clip = clip.load(
        "ViT-B/32",
        device=model_pl.device,
    )

    source_img = Image.open(args.source_img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(size=256),
            transforms.CenterCrop(size=256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    preprocessed_img = preprocess(source_img).unsqueeze(0).cuda()

    time_st = time.time()
    dummy_txt_tokens = [tokutil.tokenizer.eos_token_id] * cfg.sampling.txt_max_len
    dummy_txt_mask = np.ones(16)
    txt_item = TextInputItem(dummy_txt_tokens, dummy_txt_mask)

    txts = model_pl.sample_i2t(
        source_img=preprocessed_img,
        txt=txt_item.txt.unsqueeze(0).cuda(),
        txt_mask=txt_item.txt_mask.unsqueeze(0).cuda(),
    )

    rank = clip_score(
        txts,
        [source_img for i in range(len(txts))],
        model_clip,
        preprocess_clip,
        "cuda",
    )
    caption = txts[rank[0]]

    time_end = time.time()

    print(f"{caption}: {time_end - time_st:.02f} secs")
