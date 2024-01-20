import os
from typing import Any, List
import torch
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
from tqdm import tqdm

from magvlt.datamodules import build_datamodule, build_transform
from magvlt.datamodules.datasets.dataclass import Items

from magvlt.utils.config import build_config
from magvlt.models import build_model
from magvlt.models.utils import token2txt

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
#from magvlt.utils.fid.fid import FrechetInceptionDistance
#from magvlt.utils.fid.inception_score import InceptionScore

import clip
import numpy as np
from PIL import Image
import pickle
import glob
import math


def save_pickle(fname, data):
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


def load_fake_samples(cfg):
    fnames = glob.glob(os.path.join(cfg.result_file_path, '*.pkl'))
    pixelss = []
    for fname in tqdm(fnames, total=len(fnames)):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        data_ = np.concatenate(data['pixelss'], axis=0)
        data_ = torch.from_numpy(data_).to(dtype=torch.uint8)
        pixelss.append(data_)
    pixelss = torch.cat(pixelss, dim=0)

    return pixelss


class T2IEvalWrapper(LightningModule):
    def __init__(self, cfg, t2i_model):
        super().__init__()
        self.cfg = cfg
        self.t2i_model = t2i_model
        self.outputss = []
        self.txtss = []
        self.it_idx = 0
        self.model_clip, self.preprocess_clip = clip.load("ViT-B/32", device=model_pl.device,
                                                          download_root='/data/karlo-research_715/karlo-backbone/magvlt/gits/CLIP/pretrained')

    @torch.no_grad()
    def clip_score(self, texts, pixels, model_ar, model_vqvae, model_clip, preprocess_clip, device):
        pixels = pixels.cpu().numpy()
        pixels = np.transpose(pixels, (0, 2, 3, 1))

        images = [preprocess_clip(Image.fromarray((pixel * 255).astype(np.uint8))) for pixel in pixels]
        images = torch.stack(images, dim=0).to(device=device)
        texts = clip.tokenize(texts).to(device=device)
        texts = torch.repeat_interleave(texts, pixels.shape[0], dim=0)

        image_features = model_clip.encode_image(images)
        text_features = model_clip.encode_text(texts)

        scores = F.cosine_similarity(image_features, text_features).squeeze()
        rank = torch.argsort(scores, descending=True)

        return rank

    def token2txt(self, output_ids, tokenizer):
        eos_token_id = tokenizer.eos_token_id
        outs = []
        for output in output_ids:
            out = output.tolist()
            if eos_token_id in out:
                out = out[:out.index(eos_token_id)]
            outs.append(out)

        if callable(tokenizer):
            outputs = []
            for output in output_ids:
                outputs.append(tokenizer.decode(output.tolist(), skip_special_tokens=True))
        else:
            outputs = tokenizer.decode_batch(output_ids.tolist(), skip_special_tokens=True)
        return outputs

    def test_step(self, batch: Items, batch_idx: int):
        with torch.no_grad():
            txt_rep = torch.repeat_interleave(batch.txt, self.cfg.sampling.img_num_cand_samples, dim=0)
            txt_mask_rep = torch.repeat_interleave(batch.txt_mask, self.cfg.sampling.img_num_cand_samples, dim=0)

            pixels = self.t2i_model.sample_t2i(
                txt=txt_rep,
                txt_mask=txt_mask_rep,
                ctx_len_img=self.cfg.stage2.hparams.ctx_len_img,
                n_steps=self.cfg.stage2.mask_hparams.t2i_n_steps,
                strategy=self.cfg.sampling.img_mask_sample_method,
                temp_st=self.cfg.sampling.img_temperature_start,
                temp_end=self.cfg.sampling.img_temperature_end,
                multi_temp_st=self.cfg.sampling.img_mult_temperature_start,
                multi_temp_end=self.cfg.sampling.img_mult_temperature_end,
                return_history=False,
            )

            hh = int(math.sqrt(pixels.shape[1]))
            pixels = pixels.view(pixels.shape[0], hh, hh)
            pixels = self.t2i_model.model_vqvae.decode_code(pixels) * 0.5 + 0.5
            pixels = torch.clamp(pixels, 0, 1)
            pixels = pixels.view(batch.img.shape[0], self.cfg.sampling.img_num_cand_samples, 3, 256, 256)

            outputs = []
            # texts_dec = self.tokenizer.decode_batch(ids.tolist(), skip_special_tokens=True)
            texts_dec = token2txt(batch.txt, self.t2i_model.tokenizer)

            for j in range(batch.img.shape[0]):
                texts_dec_, pixels_ = texts_dec[j], pixels[j]
                rank = self.clip_score(texts_dec_, pixels_, self.t2i_model.model, self.t2i_model.model_vqvae,
                                       self.model_clip,
                                       self.preprocess_clip, 'cuda')
                outputs_ = [pixels_[0].detach().clone(), pixels_[rank][0].detach().clone()]
                outputs.append(torch.stack(outputs_, dim=0))
            outputs = torch.stack(outputs, dim=0)

        txts = token2txt(batch.txt, self.t2i_model.tokenizer)

        self.outputss.append((outputs.cpu().numpy() * 255.).astype(np.uint8))
        self.txtss += txts

        self.it_idx += 1

        if self.it_idx % 100 == 0:
            save_pickle(
                os.path.join(self.cfg.result_file_path, f'samples_rank_{self.global_rank}_iter_{self.it_idx}.pkl'),
                {'pixelss': self.outputss,
                 'txtss': self.txtss})

            self.outputss = []
            self.txtss = []

        return

    def on_test_epoch_end(self):
        if len(self.outputss) > 0:
            save_pickle(os.path.join(self.cfg.result_file_path, f'samples_rank_{self.global_rank}_iter_last.pkl'),
                        {'pixelss': self.outputss,
                         'txtss': self.txtss})
        print(f'done file dump at {self.cfg.result_file_path}')


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cfg, cfg_yaml = build_config()

    os.makedirs(cfg.result_file_path, exist_ok=True)

    cfg.dataset.eval_center_crop = True

    total_gpus = cfg.dist.n_gpus * cfg.dist.n_nodes

    seed_everything(cfg.sampling.seed, workers=True)

    val_transform = build_transform(
        cfg=cfg,
        split="val",
    )

    datamodule = build_datamodule(
        cfg=cfg,
        train_transform=val_transform,
        val_transform=val_transform,
        pin_memory=False,
        epoch=0,
        total_gpus=total_gpus,
    )

    n_data_length = 30000
    datamodule.setup()

    test_dataloader = datamodule.val_dataloader()
    test_dataloader.dataset.set_custom_length(n_data_length)

    if cfg.dataset.name != 'imagenet':
        cfg.stage2.vocab_size_txt = test_dataloader.dataset.get_vocab_size()

    from pytorch_lightning.strategies import DDPStrategy

    trainer = pl.Trainer(
        accelerator="gpu",
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        strategy=DDPStrategy(
            find_unused_parameters=False,
        ),
        max_epochs=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        deterministic=False,
    )

    model_pl = build_model(cfg, tokenizer=test_dataloader.dataset.tokenizer)
    model_pl.load_model(cfg.checkpoint_path)
    model_pl.eval()
    model_pl.cuda()

    wrapper = T2IEvalWrapper(cfg, model_pl)
    trainer.test(wrapper, dataloaders=test_dataloader)
    trainer.strategy.barrier()

    torch.distributed.destroy_process_group()

    if trainer.global_rank == 0:
        fake_samples = load_fake_samples(cfg)
        datamodule = build_datamodule(
            cfg=cfg,
            train_transform=None,
            val_transform=val_transform,
            pin_memory=False,
            epoch=0,
            total_gpus=1,
        )

        datamodule.setup()
        test_dataloader = datamodule.val_dataloader()
        test_dataloader.dataset.set_custom_length(n_data_length)

        imgss = []
        for it, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            imgs = ((batch.img * 0.5 + 0.5).cpu().numpy() * 255.).astype(np.uint8)
            imgss.append(torch.from_numpy(imgs).to(dtype=torch.uint8))
        imgss = torch.cat(imgss, dim=0)

        real_samples = imgss
        r_idx = np.random.permutation(
            min(fake_samples.shape[0], real_samples.shape[0])
        )[:n_data_length]

        real_samples = real_samples[r_idx]
        fake_samples = fake_samples[r_idx]

        for rank in range(2):
            fid = FrechetInceptionDistance().cuda()
            fid_batch_size = 100
            n_batches = min(math.ceil(fake_samples.shape[0] / fid_batch_size),
                            math.ceil(real_samples.shape[0] / fid_batch_size))
            for i in tqdm(range(n_batches), total=n_batches):
                sp = i * fid_batch_size
                ep = (i + 1) * fid_batch_size

                real_samples_ = real_samples[sp:ep]
                fake_samples_ = fake_samples[:, rank][sp:ep]

                fid.update(real_samples_.cuda(), real=True)
                fid.update(fake_samples_.cuda(), real=False)

            fid_score = fid.compute()

            if rank == 0:
                print(cfg.result_file_path)
                print("No CLIP Re-ranking, FID: %.4f" % fid_score)
            else:
                print("CLIP Re-ranking, FID: %.4f" % fid_score)

        for rank in range(2):
            incs = InceptionScore().cuda()
            incs_batch_size = 100
            n_fake_batches = math.ceil(fake_samples.shape[0] / incs_batch_size)
            for i in tqdm(range(n_fake_batches), total=n_fake_batches):
                sp = i * incs_batch_size
                ep = (i + 1) * incs_batch_size
                fake_samples_ = fake_samples[:, rank][sp:ep]
                incs.update(fake_samples_.cuda())

            inception_score = incs.compute()

            if rank == 0:
                print(f"No CLIP Re-ranking, IS mean: {inception_score[0]:.4f}, IS std: {inception_score[1]:.4f}")
            else:
                print(f"CLIP Re-ranking, IS mean: {inception_score[0]:.4f}, IS std: {inception_score[1]:.4f}")

    print('done!')