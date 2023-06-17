import argparse
import time

import clip
import torch
import torchvision.transforms as transforms
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from PIL import Image, ImageFilter

from magvlt.datamodules.tokenizers import TokenizerUtils
from magvlt.models import build_model
from magvlt.models.utils import token2txt, clip_score


def default_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        default="a male tennis player in white shorts is playing tennis",
        help="The prompt to guide the image generation",
    )
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


def gen_mask():
    mask_center_yx = (8, 8)
    mask = torch.ones(16, 16).unsqueeze(0).cuda()
    mask_size = 8  # 50% center masking
    mask_y_min = max(0, mask_center_yx[0] - mask_size // 2)
    mask_y_max = min(16, mask_center_yx[0] + mask_size // 2)
    mask_x_min = max(0, mask_center_yx[1] - mask_size // 2)
    mask_x_max = min(16, mask_center_yx[1] + mask_size // 2)
    mask[:, mask_y_min:mask_y_max, mask_x_min:mask_x_max] = 0

    composit_mask = Image.fromarray(np.uint8(mask[0].cpu() * 255.0))
    composit_mask = composit_mask.resize((256, 256))
    composit_mask = composit_mask.filter(ImageFilter.GaussianBlur(radius=15))
    composit_mask = np.float32(composit_mask)[:, :, np.newaxis] / 255.0
    mask_256_ = torch.tensor(composit_mask).squeeze().cuda()
    return mask, mask_256_


if __name__ == "__main__":
    parser = default_parser()
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_path)
    cfg.stage2.mask_hparams.t2i_n_steps = cfg.sampling.img_num_steps

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

    gt_txt = args.prompt
    source_img = Image.open(args.source_img_path)
    transforms_ = transforms.Compose(
        [
            transforms.Resize(size=256),
            transforms.CenterCrop(size=256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    source_img = transforms_(source_img).cuda().unsqueeze(0)
    img_mask, img_mask_256 = gen_mask()

    time_st = time.time()
    txt_item = tokutil.get_input(gt_txt)
    txt = txt_item.txt.unsqueeze(0).cuda()
    txt_mask = txt_item.txt_mask.unsqueeze(0).cuda()
    txt_rep = torch.repeat_interleave(txt, cfg.sampling.img_num_cand_samples, dim=0)
    txt_mask_rep = torch.repeat_interleave(
        txt_mask, cfg.sampling.img_num_cand_samples, dim=0
    )

    pixels = model_pl.sample_it2i(
        txt=txt_rep,
        txt_mask=txt_mask_rep,
        ctx_len_img=cfg.stage2.hparams.ctx_len_img,
        n_steps=cfg.stage2.mask_hparams.t2i_n_steps,
        strategy=cfg.sampling.img_mask_sample_method,
        temp_st=cfg.sampling.img_temperature_start,
        temp_end=cfg.sampling.img_temperature_end,
        multi_temp_st=cfg.sampling.img_mult_temperature_start,
        multi_temp_end=cfg.sampling.img_mult_temperature_end,
        source_img=source_img,
        img_mask=img_mask,
    )

    txt_desc = token2txt(txt, tokutil.tokenizer)

    rank = clip_score(
        [txt_desc[0] for _ in range(cfg.sampling.img_num_cand_samples)],
        pixels,
        model_clip,
        preprocess_clip,
        "cuda",
    )
    time_end = time.time()

    plt.figure(figsize=(4, 4))
    plt.imshow(pixels[rank[0]])

    plt.axis("off")
    plt.tight_layout()
    plt.show()
    print(f"{gt_txt}: {time_end-time_st:.02f} secs")
