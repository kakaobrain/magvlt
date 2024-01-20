import os
from typing import Any, List
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, seed_everything
import json

from magvlt.datamodules import build_datamodule, build_transform

from magvlt.utils.config import build_config
from magvlt.models import build_model
from magvlt.models.utils import token2txt
from magvlt.utils.sampling import sample_1d_fast_i2t
from magvlt.datamodules.tokenizers import TokenizerUtils
from magvlt.datamodules.datasets.dataclass import Items
from magvlt.models.maskgit.maskgit import MaskGITModel

import clip
import numpy as np
from PIL import Image



class I2TEvalWrapper(LightningModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg, tokenizer=tokenizer)
        # attn plus
        attn_plus = getattr(cfg.experiment, 'attn_plus', None)
        self.model.model.attn_plus = attn_plus
        self.tokenizer = tokenizer
        self.model_clip = None
        self.preprocess_clip = None

        self.task = 'caption'
        if self.cfg.dataset.name.startswith('vqa'):
            self.task = 'vqa'

        self.results = []

    def load(self):
        assert self.cfg.checkpoint_path != 'none'
        ckpt = torch.load(self.cfg.checkpoint_path, map_location='cpu')
        self.model.load_stage1_model(self.cfg.stage1.path)
        #load_stage1_ckpt(self.model.model_vqvae, self.cfg.stage1.path)
        if self.cfg.experiment.strategy.type == 'zero1':
            for key in ckpt['module'].copy().keys():
                new_key = key[7:]
                ckpt['module'][new_key] = ckpt['module'].pop(key)
            self.load_state_dict(ckpt['module'], strict=True)
        else:
            self.model.load_state_dict(ckpt['state_dict'], strict=True)
        self.model_clip, self.preprocess_clip = clip.load("ViT-B/32", device=self.device)

    @torch.no_grad()
    def clip_reranking(self, image, prompts):
        image = self.preprocess_clip(Image.fromarray((image[0] * 255).astype(np.uint8))).cuda().unsqueeze(0)
        texts = [clip.tokenize([prompt]) for prompt in prompts]
        texts = torch.stack(texts, dim=0).cuda()

        image_feature = self.model_clip.encode_image(image)
        text_features = [self.model_clip.encode_text(text) for text in texts]
        text_features = torch.cat(text_features, dim=0).cuda()

        scores = F.cosine_similarity(image_feature, text_features).cpu().numpy().squeeze()
        ranked = np.argsort(scores)[::-1]

        # print("CLIP scores (sorted by descending order): ", scores[ranked])
        return ranked

    def generate_text(self, batch: Items):
        sample_method = self.cfg.sampling.txt_sample_method
        max_seq_len = self.cfg.sampling.txt_max_len
        top_k = self.cfg.sampling.txt_top_k
        top_p = self.cfg.sampling.txt_top_p
        temperature = self.cfg.sampling.txt_temperature
        if top_p == 1:
            top_p = None
        eos_token_id = 0

        images = batch.img.cuda()
        with torch.no_grad():
            cond_img = self.model.model_vqvae.get_codes(images).detach()
        cond_txt = None
        txt_pos_id = batch.txt_pos_id.clip(0)
        tokenizer = self.tokenizer

        if hasattr(tokenizer, 'pad_token_id'):
            eos_token_id = tokenizer.pad_token_id

        if isinstance(self.model, MaskGITModel):
            batch_size = images.shape[0]
            num_steps = self.cfg.sampling.txt_num_cand_samples // self.cfg.sampling.txt_sample_size
            num_sample_per_step = self.cfg.sampling.txt_sample_size
            num_candidates = self.cfg.sampling.txt_num_cand_samples

            texts_token, score = [], []
            for _ in range(num_steps):
                _texts_token, _score = self.model.sample_i2t(batch, tok_img=cond_img, sample_size=num_sample_per_step, return_tokens=True)
                texts_token.append(_texts_token)
                score.append(_score)
            texts_token = torch.cat(texts_token, dim=0)
            score = torch.cat(score, dim=0)

            # scoring
            index = torch.argmax(score.view(batch_size, num_candidates), dim=-1) + torch.arange(batch_size).to(score.device) * num_candidates
            _texts_token = texts_token.index_select(0, index)
            texts = token2txt(_texts_token, tokenizer)
        else:
            if sample_method in ['sample', 'argmax']:
                texts_token = sample_1d_fast_i2t(self.model, cond_img, temperature=temperature, top_k=top_k, top_p=top_p, amp=True, max_seq_len=max_seq_len, eos_token_id=eos_token_id, txt=cond_txt)
                texts = token2txt(texts_token, tokenizer)
            elif sample_method == 'rerank':
                # TODO: implement clip-rerank
                with torch.no_grad():
                    cond_img = torch.repeat_interleave(cond_img, self.cfg.sampling.txt_sample_size, dim=0)
                    texts_token_list = []
                    for _ in range(self.cfg.sampling.txt_num_cand_samples // self.cfg.sampling.txt_sample_size):
                        texts_token = sample_1d_fast_i2t(self.model, cond_img, temperature=temperature, top_k=top_k, top_p=top_p, amp=True, max_seq_len=max_seq_len, eos_token_id=eos_token_id, txt=cond_txt)
                        texts_token = F.pad(texts_token, [0, max_seq_len-texts_token.shape[-1]], mode='constant', value=eos_token_id)
                        texts_token_list.append(texts_token)
                    texts_token = torch.cat(texts_token_list)


        if 'rerank' in sample_method:
            if sample_method == 'rerank':
                texts_token = texts_token.view(cond_img.size(0), self.cfg.sampling.txt_num_cand_samples // self.cfg.sampling.txt_sample_size, -1)
            texts_token = torch.split(texts_token, self.cfg.sampling.txt_sample_size, dim=0)
            images = torch.split(images, 1, 0)

            texts = []
            for text_token, image in zip(texts_token, images):
                image = image * 0.5 + 0.5
                image = torch.clamp(image, 0, 1).cpu().numpy()
                image = np.transpose(image, (0, 2, 3, 1))
                cand_texts = token2txt(text_token.view(-1, text_token.size(-1)), tokenizer)
                ranked = self.clip_reranking(image, cand_texts)
                cand_texts = np.array(cand_texts)[ranked]
                texts.append(cand_texts[0])

        return texts

    def test_step(self, batch: Items, batch_idx: int):
        cond_txt = None
        if self.task == 'caption':
            identifier = batch.imgpath
        elif self.task == 'vqa':
            identifier = batch.id
            cond_txt = batch.txt

        images = batch.img.cuda()
        #result = []
        with torch.no_grad():
            # texts = self.generate_text(img_codes, images, self.tokenizer, cond_txt=cond_txt, txt_pos_id=batch.txt_pos_id.clip(min=0))
            texts = self.generate_text(batch)
        B = len(images)

        if self.task == 'caption':
            captions = batch.gt_txt
            for i in range(B):
                captions_i = captions[i]
                captions_i = [TokenizerUtils.pre_caption(caption) for caption in captions_i]
                dict_ = {
                    "split": "test",
                    "image_name": identifier[i],
                    "captions": captions_i,
                    "prediction": texts[i]
                }
                if cfg.dataset.name == 'nocaps':
                    dict_["domain"] = batch.domain[i]
                self.results.append(dict_)
        elif self.task == 'vqa':
            for i in range(B):
                dict_ = {
                    "question_id": identifier[i],
                    "answer": texts[i]
                }
                self.results.append(dict_)

        return self.results

    def on_test_epoch_end(self):
        if cfg.dataset.name.startswith('coco'):
            # outputs_ = []
            # for output in self.results:
            #     outputs_.extend(output)
            outputs_ = self.results
            # print(outputs_[0])
            fname, ext = os.path.splitext(self.cfg.result_file_path)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            result_file_path = f'{fname}_{self.global_rank}{ext}'
            with open(result_file_path, 'w') as outfile:
                json.dump(outputs_, outfile)
            print(f'done json file dump at {result_file_path}')

        elif cfg.dataset.name == 'nocaps':
            outputs_all = []
            outputs_in = []
            outputs_near = []
            outputs_out = []
            for output in self.results:
                for out in output:
                    outputs_all.append(out)
                    if out["domain"] == 'in-domain':
                        outputs_in.append(out)
                    elif out["domain"] == 'near-domain':
                        outputs_near.append(out)
                    elif out["domain"] == 'out-domain':
                        outputs_out.append(out)
                    else:
                        raise NotImplementedError
            fname, ext = os.path.splitext(self.cfg.result_file_path)
            os.makedirs(os.path.dirname(fname), exist_ok=True)

            result_file_path = f'{fname}_all_{self.global_rank}{ext}'
            with open(result_file_path, 'w') as outfile:
                json.dump(outputs_all, outfile)
            print(f'done json file dump at {result_file_path}')

            result_file_path = f'{fname}_in_{self.global_rank}{ext}'
            with open(result_file_path, 'w') as outfile:
                json.dump(outputs_in, outfile)
            print(f'done json file dump at {result_file_path}')

            result_file_path = f'{fname}_near_{self.global_rank}{ext}'
            with open(result_file_path, 'w') as outfile:
                json.dump(outputs_near, outfile)
            print(f'done json file dump at {result_file_path}')

            result_file_path = f'{fname}_out_{self.global_rank}{ext}'
            with open(result_file_path, 'w') as outfile:
                json.dump(outputs_out, outfile)
            print(f'done json file dump at {result_file_path}')



if __name__ == "__main__":
    cfg, cfg_yaml = build_config()

    assert cfg.dataset.name in ['coco2014', 'nocaps']

    seed_everything(cfg.sampling.seed, workers=True)

    val_transform = build_transform(
        cfg=cfg,
        split="val",
    )

    datamodule = build_datamodule(
        cfg=cfg,
        train_transform=None,
        val_transform=val_transform,
        pin_memory=False,
        epoch=0,
        total_gpus=cfg.dist.n_gpus,
    )

    datamodule.setup()
    if cfg.dataset.name.startswith("coco"):
        test_dataloader = datamodule.test_dataloader()
        test_dataloader.dataset.set_custom_length(10)
    elif cfg.dataset.name == 'nocaps':
        test_dataloader = datamodule.val_dataloader()
        test_dataloader.dataset.set_custom_length(4500)
    else:
        raise NotImplementedError

    cfg.stage2.vocab_size_txt = test_dataloader.dataset.get_vocab_size()

    trainer = pl.Trainer(
        accelerator="gpu",
        num_nodes=cfg.dist.n_nodes,
        devices=cfg.dist.n_gpus,
        max_epochs=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_test_batches=1.0,
        deterministic=True
    )

    wrapper = I2TEvalWrapper(cfg, test_dataloader.dataset.tokenizer)
    wrapper.load()
    wrapper.eval()

    trainer.test(wrapper, dataloaders=test_dataloader)
    trainer.strategy.barrier()

    if trainer.global_rank == 0:
        from glob import glob
        head, tail = os.path.splitext(cfg.result_file_path)
        n_files = cfg.dist.n_gpus * cfg.dist.n_nodes

        if cfg.dataset.name.startswith('coco'):
            files = glob(head + "_*" + tail)

            assert len(files) == n_files

            total_data = []
            for file in files:
                with open(file, 'r') as fin:
                    total_data.extend(json.load(fin))

            print("Number of Generated Samples:", len(total_data))
            with open(cfg.result_file_path, 'w') as fout:
                json.dump(total_data, fout)

            for file in files:
                os.remove(file)

            if wrapper.task == 'caption':
                os.system(f'python ./evaluation/cocoeval.py --result_file_path={cfg.result_file_path}')
            elif wrapper.task == 'vqa':
                os.system(f'python ./evaluation/vqa_eval.py --result_file_path={cfg.result_file_path}')

        elif cfg.dataset.name == 'nocaps':
            domain = ['in', 'near', 'out', 'all']

            for d in domain:
                print(f"----{d}-domain-----")

                files = glob(head + f"_{d}_*" + tail)

                assert len(files) == n_files

                total_data = []
                for file in files:
                    with open(file, 'r') as fin:
                        total_data.extend(json.load(fin))

                print("Number of Generated Samples:", len(total_data))
                result_file_path = head + f"_{d}" + tail
                with open(result_file_path, 'w') as fout:
                    json.dump(total_data, fout)

                for file in files:
                    os.remove(file)

                if wrapper.task == 'caption':
                    os.system(f'python ./evaluation/cocoeval.py --result_file_path={result_file_path}')
                elif wrapper.task == 'vqa':
                    os.system(f'python ./evaluation/vqa_eval.py --result_file_path={result_file_path}')


