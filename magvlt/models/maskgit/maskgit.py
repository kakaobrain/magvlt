import math

import numpy as np
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast
from pytorch_lightning import LightningModule
from PIL import Image

from magvlt.models.utils import get_positional_enc, token2txt, top_k_top_p_filtering
from magvlt.datamodules.datasets.dataclass import Items


eps = torch.finfo(torch.float16).eps


class MaskGITModel(LightningModule):
    def __init__(self, cfg, model, model_vqvae, tokenizer):
        LightningModule.__init__(self)
        self.save_hyperparameters(
            ignore=["model_vqvae", "model", "cfg"]
        )  # TODO hyperparam check
        self.model = model
        self.model_vqvae = model_vqvae
        self.tokenizer = tokenizer
        self.cfg = cfg

        self.task_mode = getattr(self.cfg.stage2.mask_hparams, "task", "it2it-task")
        tasks = getattr(self.cfg.stage2.mask_hparams, "tasks", "t2i,i2t,it2it-mask")
        self.tasks = tasks.split(",")

        self.use_spc_pos = getattr(self.cfg.stage2.hparams, "use_spc_pos", False)
        self.use_unroll_loss = getattr(
            self.cfg.stage2.hparams, "use_unroll_loss", False
        )

        self.use_cond_unroll_mask = getattr(
            self.cfg.stage2.hparams, "use_cond_unroll_mask", False
        )
        self.use_length_predictor = not self.task_mode in ["it2l"]
        self.use_hnh_task = getattr(self.cfg.dataset, "use_hnh_task", False)

        # downstream mode
        if getattr(self.cfg.dataset, "name", None) in ["vqav2", "coco2014"]:
            self.use_hnh_task = False

        self.hnh_task_scheme = getattr(self.cfg.dataset, "hnh_task_scheme", 0)
        self.use_cutmix_loss = getattr(
            self.cfg.stage2.hparams, "use_cutmix_loss", False
        )

        self.should_sample_task = self.task_mode in ["it2it-task", "it2it-mask"]

        self.ctx_len_img = cfg.stage2.hparams.ctx_len_img
        self.ctx_len_txt = cfg.stage2.hparams.ctx_len_txt
        self.val_tasks = []

        if self.task_mode == "i2t" or self.task_mode == "it2l":
            self.task_weights = [0, 1, 0]
        elif self.task_mode == "t2i":
            self.task_weights = [1, 0, 0]
        elif self.task_mode == "it2it-task":
            task_weights = getattr(
                self.cfg.stage2.mask_hparams, "task_weights", "1,1,0"
            )
            self.task_weights = list(map(int, task_weights.split(",")))
        elif self.task_mode == "it2it-mask":
            self.task_weights = [0, 0, 1]
        else:
            raise ValueError(f"{self.task_mode} is invalid task...")
        val_tasks = ["t2i", "i2t"]
        for i, w in enumerate(self.task_weights[:2]):
            if w > 0:
                self.val_tasks.append(val_tasks[i])

        self.bot_idx = None
        if self.use_length_predictor:
            self.length_loss_coef = getattr(
                self.cfg.stage2.hparams, "length_loss_coef", 1.0
            )
            self.bot_idx = 2 + self.ctx_len_img + 1
            self.model.length_predictor = torch.nn.Linear(
                cfg.stage2.hparams.embed_dim, self.ctx_len_txt, bias=False
            )
            self.model.length_predictor.weight.data.normal_(mean=0.0, std=0.02)

        if self.task_mode == "it2l":
            self.q_token_idx = 2 + self.ctx_len_img + 2
            dropout_rate = getattr(self.cfg.stage2.mask_hparams, "drop_rate", 0.1)
            self.dropout = torch.nn.Dropout(dropout_rate)
            classifier = torch.nn.Linear(
                cfg.stage2.hparams.embed_dim, self.cfg.stage2.mask_hparams.n_class
            )
            self.model.classifier = classifier
            self.classifier = classifier
        if self.use_hnh_task:
            self.hnh_loss_coef = getattr(self.cfg.stage2.hparams, "hnh_loss_coef", 0.5)

        self.sep_token_id = self.tokenizer.token_to_id("|")

    def gen_t2i_tok_spc(self, batch_size, device, is_first=None):
        tok_BOC = (
            torch.IntTensor([[self.model.spc_tok_class.BOC]])
            .to(device)
            .repeat_interleave(batch_size, dim=0)
        )
        tok_BOG = (
            torch.IntTensor([[self.model.spc_tok_class.BOG]])
            .to(device)
            .repeat_interleave(batch_size, dim=0)
        )
        tok_BOT = (
            torch.IntTensor([[self.model.spc_tok_class.BOT]])
            .to(device)
            .repeat_interleave(batch_size, dim=0)
        )
        tok_BOI = (
            torch.IntTensor([[self.model.spc_tok_class.BOI]])
            .to(device)
            .repeat_interleave(batch_size, dim=0)
        )

        if self.use_hnh_task and is_first is not None:
            tok_BOG_left = (
                torch.IntTensor([[self.model.spc_tok_class.BOG_LEFT]])
                .to(device)
                .repeat_interleave(batch_size, dim=0)
            )
            tok_BOG_right = (
                torch.IntTensor([[self.model.spc_tok_class.BOG_RIGHT]])
                .to(device)
                .repeat_interleave(batch_size, dim=0)
            )
            tok_BOG = torch.where(is_first, tok_BOG_left, tok_BOG_right)

        return tok_BOC, tok_BOG, tok_BOT, tok_BOI

    #
    def gen_i2t_tok_spc(self, batch_size, device, is_first=None, is_vertical=True):
        tok_BOC = (
            torch.IntTensor([[self.model.spc_tok_class.BOC]])
            .to(device)
            .repeat_interleave(batch_size, dim=0)
        )
        tok_BOG = (
            torch.IntTensor([[self.model.spc_tok_class.BOG]])
            .to(device)
            .repeat_interleave(batch_size, dim=0)
        )
        tok_BOT = (
            torch.IntTensor([[self.model.spc_tok_class.BOT]])
            .to(device)
            .repeat_interleave(batch_size, dim=0)
        )
        tok_BOI = (
            torch.IntTensor([[self.model.spc_tok_class.BOI]])
            .to(device)
            .repeat_interleave(batch_size, dim=0)
        )

        if self.use_hnh_task and not self.use_cutmix_loss:
            if is_first is not None:
                if not is_vertical:
                    tok_BOG_left = (
                        torch.IntTensor([[self.model.spc_tok_class.BOG_LEFT]])
                        .to(device)
                        .repeat_interleave(batch_size, dim=0)
                    )
                    tok_BOG_right = (
                        torch.IntTensor([[self.model.spc_tok_class.BOG_RIGHT]])
                        .to(device)
                        .repeat_interleave(batch_size, dim=0)
                    )
                    tok_BOG = torch.where(is_first, tok_BOG_left, tok_BOG_right)
                else:
                    tok_BOG_top = (
                        torch.IntTensor([[self.model.spc_tok_class.BOG_TOP]])
                        .to(device)
                        .repeat_interleave(batch_size, dim=0)
                    )
                    tok_BOG_bottom = (
                        torch.IntTensor([[self.model.spc_tok_class.BOG_BOTTOM]])
                        .to(device)
                        .repeat_interleave(batch_size, dim=0)
                    )
                    tok_BOG = torch.where(is_first, tok_BOG_top, tok_BOG_bottom)

        return tok_BOC, tok_BOG, tok_BOT, tok_BOI

    def _get_img_locs(self, task, B, T, img_locs, device):
        txt_locs = torch.zeros([B, T], device=device)
        if task == "t2i":
            locs = [
                torch.zeros([B, 2], device=device),
                txt_locs,
                torch.zeros([B, 2], device=device),
                img_locs,
            ]
        else:
            locs = [
                torch.zeros([B, 2], device=device),
                img_locs,
                torch.zeros([B, 2], device=device),
                txt_locs,
            ]
        return torch.cat(locs, dim=1).bool().to(device)

    def _get_txt_locs(self, task, B, I, txt_locs, device):
        img_locs = torch.zeros([B, I], device=device)
        if task == "t2i":
            locs = [
                torch.zeros([B, 2], device=device),
                txt_locs,
                torch.zeros([B, 2], device=device),
                img_locs,
            ]
        else:
            locs = [
                torch.zeros([B, 2], device=device),
                img_locs,
                torch.zeros([B, 2], device=device),
                txt_locs,
            ]
        return torch.cat(locs, dim=1).bool().to(device)

    def _get_spc_locs(self, task, B, T, I, device):
        img_locs = torch.zeros([B, I], device=device)
        txt_locs = torch.zeros([B, T], device=device)
        if task == "t2i":
            locs = [
                torch.ones([B, 2], device=device),
                txt_locs,
                torch.ones([B, 2], device=device),
                img_locs,
            ]
        else:
            locs = [
                torch.ones([B, 2], device=device),
                img_locs,
                torch.ones([B, 2], device=device),
                txt_locs,
            ]
        return torch.cat(locs, dim=1).bool().to(device)

    def _get_input(
        self, tok_img, tok_txt, task=None, pos_txt=None, is_first=None, is_vertical=True
    ):
        device = tok_img.device
        B = tok_img.shape[0]
        T = tok_txt.shape[1]
        I = tok_img.shape[1]

        if pos_txt is None:
            pos_txt = get_positional_enc(tok_txt)
        pos_img = get_positional_enc(tok_img)
        if self.use_spc_pos:
            prefix_pos_spc = torch.tensor([[0, 1]]).int().to(device).repeat([B, 1])
            infix_pos_spc = torch.tensor([[2, 3]]).int().to(device).repeat([B, 1])
        else:
            prefix_pos_spc = torch.zeros([B, 2]).int().to(device)
            infix_pos_spc = torch.zeros([B, 2]).int().to(device)

        if task == "t2i":
            tok_BOC, tok_BOG, tok_BOT, tok_BOI = self.gen_t2i_tok_spc(
                batch_size=B, device=device, is_first=is_first
            )
            tok = torch.cat(
                [tok_BOC, tok_BOT, tok_txt, tok_BOG, tok_BOI, tok_img], dim=1
            )
            img_locs = self._get_img_locs(
                task, B, T, torch.ones([B, I], device=device), device
            )
            txt_locs = self._get_txt_locs(
                task, B, I, torch.ones([B, T], device=device), device
            )
            spc_locs = self._get_spc_locs(task, B, T, I, device)
            pos = torch.cat([prefix_pos_spc, pos_txt, infix_pos_spc, pos_img], dim=1)
        elif task in ["i2t", "it2it-mask"]:
            tok_BOC, tok_BOG, tok_BOT, tok_BOI = self.gen_i2t_tok_spc(
                batch_size=B, device=device, is_first=is_first, is_vertical=is_vertical
            )
            tok = torch.cat(
                [tok_BOC, tok_BOI, tok_img, tok_BOG, tok_BOT, tok_txt], dim=1
            )
            img_locs = self._get_img_locs(
                task, B, T, torch.ones([B, I], device=device), device
            )
            txt_locs = self._get_txt_locs(
                task, B, I, torch.ones([B, T], device=device), device
            )
            spc_locs = self._get_spc_locs(task, B, T, I, device)
            pos = torch.cat([prefix_pos_spc, pos_img, infix_pos_spc, pos_txt], dim=1)
        else:
            raise ValueError(f"{task} is invalid task...")

        return tok, pos, img_locs, txt_locs, spc_locs

    def _mask_schedule(self, r, task=None):
        schedule = None
        if task == "t2i":
            schedule = self.cfg.stage2.mask_hparams.t2i_schedule
        elif task == "i2t":
            schedule = self.cfg.stage2.mask_hparams.i2t_schedule
        elif task == "it2it-mask":
            return torch.stack(
                [self._mask_schedule(r, task="t2i"), self._mask_schedule(r, task="i2t")]
            )
        if schedule == "cosine":
            return torch.cos(0.5 * math.pi * r)
        elif schedule == "linear":
            return r if task == "i2t" else 1 - r
        elif schedule == "sqrt":
            return torch.sqrt(r) if task == "i2t" else 1 - torch.sqrt(r)
        else:
            raise ValueError(f"{schedule} is invalid mask schedule...")

    def _get_next_mask(
        self,
        prev_loc_mask,
        confidence,
        r,
        length,
        strategy="multinomial-maskgit",
        multi_temp=1.0,
        is_last_step=False,
    ):
        B = prev_loc_mask.shape[0]

        L = prev_loc_mask.shape[1]
        r = torch.Tensor([r] * B)
        prev_n = prev_loc_mask[0].sum(-1).cpu()
        N = (
            prev_n
            - torch.clamp(
                torch.ceil(self._mask_schedule(r, task="t2i") * length), 0, prev_n
            )
        ).int()
        if not is_last_step:
            N = torch.clamp(N, 1, None)
        N = N.tolist()

        loc_mask = prev_loc_mask.clone()
        for b, n in zip(range(B), N):
            masked_ids = torch.arange(L)[prev_loc_mask[b].cpu()]
            scores_on_masked_ids = confidence[b][masked_ids]

            masked_ids_sorted = masked_ids[
                torch.sort(scores_on_masked_ids, descending=True).indices.cpu()
            ]
            if strategy == "maskgit":
                newly_unmasked_ids = masked_ids_sorted[:n]
            elif strategy == "random":
                candidates = masked_ids_sorted
                subset = torch.randperm(candidates.shape[0])[:n]
                newly_unmasked_ids = masked_ids_sorted[subset]
            elif strategy == "multinomial":
                newly_unmasked_ids = masked_ids[
                    (scores_on_masked_ids + eps).multinomial(n)
                ]
            elif strategy == "multinomial-maskgit":
                # identical to multinomial-softmax-log
                logits = torch.log(scores_on_masked_ids.clamp(eps, 1))
                gumbels = (
                    -torch.empty_like(
                        logits, memory_format=torch.legacy_contiguous_format
                    )
                    .exponential_()
                    .log()
                )
                perturbed_logits = logits + multi_temp * gumbels
                newly_unmasked_ids = masked_ids[perturbed_logits.topk(n).indices.cpu()]
            loc_mask[b, newly_unmasked_ids] = torch.logical_not(
                loc_mask[b, newly_unmasked_ids]
            )

        return loc_mask

    def _get_i2t_attn_mask(self, B, L, T, offset, n_seq, device):
        attn_mask = torch.ones([B, L, L]).bool().to(device)
        for i, n in enumerate(n_seq):
            attn_mask[i, :, offset + n : offset + T] = False
        if self.bot_idx is not None:
            attn_mask[:, self.bot_idx, self.bot_idx + 1 :] = False

        return attn_mask

    def _get_t2i_mask(self, loc_img, r, loc_mask=None):
        B = loc_img.shape[0]
        I = loc_img[0].sum()
        offset = loc_img[0].nonzero()[0]
        if loc_mask is None:
            N = torch.clamp(torch.ceil(r * I), 1, I).int().tolist()
            loc_mask = torch.zeros_like(loc_img).bool()
            for b, n in zip(range(B), N):
                loc = torch.rand([I]).to(loc_img.device).topk(n, dim=0).indices + offset
                loc_mask[b].scatter_(dim=-1, index=loc, value=True)
        else:
            cond_loc_mask = (loc_mask.clone().detach())[
                :, offset : offset + self.ctx_len_img
            ]
            n_masks = cond_loc_mask.sum(-1).long()
            new_loc_mask = torch.zeros_like(loc_img).bool()
            for b, seq_r, mask_len in zip(range(B), r, n_masks):
                n = torch.clamp(torch.floor(seq_r * mask_len), 1, mask_len).int()
                mask_target = loc_mask[b].nonzero().squeeze()
                ids = (
                    torch.rand([mask_len])
                    .to(loc_img.device)
                    .topk(n, dim=0)
                    .indices.long()
                )
                loc = mask_target.index_select(0, ids)
                new_loc_mask[b].scatter_(dim=-1, index=loc, value=True)
            loc_mask = new_loc_mask
        return loc_mask

    def _get_i2t_mask(
        self, loc_txt, n_seqs, offset, r, loc_mask=None, hnh_loc_txt=None
    ):
        B = loc_txt.shape[0]
        if loc_mask is None:
            loc_mask = torch.zeros_like(loc_txt).bool()
        if hnh_loc_txt is None:
            for b, seq_r, seq_len in zip(range(B), r, n_seqs):
                n = torch.clamp(torch.ceil(seq_r * seq_len), 1, seq_len).int()
                loc = (
                    torch.rand([seq_len]).to(loc_txt.device).topk(n, dim=0).indices
                    + offset
                )
                loc_mask[b].scatter_(dim=-1, index=loc, value=True)
        else:
            n_seqs = hnh_loc_txt.sum(-1).long()
            for b, seq_r, seq_len in zip(range(B), r, n_seqs):
                mask_target = hnh_loc_txt[b].nonzero().squeeze()
                n = torch.clamp(torch.ceil(seq_r * seq_len), 1, seq_len).int()
                ids = (
                    torch.rand([seq_len])
                    .to(loc_txt.device)
                    .topk(n, dim=0)
                    .indices.long()
                )
                loc = mask_target.index_select(0, ids)
                loc_mask[b].scatter_(dim=-1, index=loc, value=True)

        return loc_mask

    def _get_mask(
        self,
        loc_img,
        loc_txt,
        loc_spc,
        r=None,
        task=None,
        txt_mask=None,
        loc_mask=None,
        hnh_loc_img=None,
        hnh_loc_txt=None,
    ):
        device = loc_img.device
        if task is None:
            task = self.task_mode

        B = loc_img.shape[0]
        L = loc_img.shape[1]
        T = loc_txt[0].sum()
        if r is None:
            r = torch.rand([B])
        else:
            if not isinstance(r, torch.Tensor):
                r = torch.Tensor([r])
            if r.shape[0] != B:
                r = torch.repeat_interleave(r, B, dim=0)
        r = r.to(device)
        r = self._mask_schedule(r, task=task)

        txt_n_seqs = txt_mask.sum(-1).long()
        txt_offset = loc_txt[0].nonzero()[0]
        if task == "t2i":
            loc_mask = self._get_t2i_mask(loc_img, r, loc_mask=loc_mask)
            attn_mask = torch.ones([B, L, L]).bool().to(device)
            for i, n in enumerate(txt_n_seqs):
                attn_mask[i, txt_offset + n : txt_offset + T, :] = False
                attn_mask[i, :, txt_offset + n : txt_offset + T] = False
        elif task == "i2t":
            loc_mask = self._get_i2t_mask(loc_txt, txt_n_seqs, txt_offset, r)
            attn_mask = self._get_i2t_attn_mask(B, L, T, txt_offset, txt_n_seqs, device)
        elif task == "it2it-mask":
            if hnh_loc_img is None:
                loc_mask = self._get_t2i_mask(loc_img, r[0], loc_mask=loc_mask)
            else:
                assert loc_mask is None
                loc_mask = self._get_t2i_mask(loc_img, r[0], loc_mask=hnh_loc_img)
            loc_mask = self._get_i2t_mask(
                loc_txt,
                txt_n_seqs,
                txt_offset,
                r[1],
                loc_mask=loc_mask,
                hnh_loc_txt=hnh_loc_txt,
            )
            attn_mask = self._get_i2t_attn_mask(B, L, T, txt_offset, txt_n_seqs, device)
        else:
            raise ValueError(f"{task} is invalid task..")

        return loc_mask, attn_mask, r

    def _apply_mask(self, tok, sps_locs, mask_locs, loc_img, loc_txt):
        tok_masked = tok.clone()
        tok_masked[
            torch.logical_and(mask_locs, loc_img)
        ] = self.model.spc_tok_class.MASK_IMG
        tok_masked[
            torch.logical_and(mask_locs, loc_txt)
        ] = self.model.spc_tok_class.MASK_TXT
        spc_locs = torch.logical_or(sps_locs, mask_locs)
        return tok_masked, spc_locs

    def load_stage1_model(self, ckpt_path):
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
        )["state_dict"]
        ckpt_ = {}
        for key, value in ckpt.items():
            if not key.startswith("loss"):
                ckpt_[key] = value
        self.model_vqvae.load_state_dict(ckpt_)

    def load_model(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]

        strict = True
        self.load_state_dict(state_dict, strict=strict)

    def get_logits_and_target_pair(self, x, tok_txt, loc_img, loc_txt):
        # FIXME: it would be better if this block is inside the model
        if not hasattr(self.model, "lm_layer"):
            logit_txt = self.model.head_txt(x[loc_txt])
            logit_img = self.model.head_img(x[loc_img])
        else:
            logits = self.model.lm_layer(x, self.model.tok_emb)
            logit_txt = logits[loc_txt]
            logit_img = logits[loc_img]
            if not self.model.use_target_vocab_only:
                tok_txt += self.model.vocab_size_img
            else:
                logit_txt = logit_txt[:, self.model.vocab_size_img :]
                logit_img = logit_img[:, : self.model.vocab_size_img]
        return logit_txt, logit_img, tok_txt

    def predict_length(self, model_output):
        h = model_output[:, self.bot_idx, :]
        length_logits = self.model.length_predictor(h).view([h.shape[0], -1])
        return length_logits

    def _get_txt_mask(self, loc_txt, log_probs, N, n, use_topk=True):
        offset = loc_txt[0].nonzero()[0]
        loc_mask = torch.zeros_like(loc_txt).bool()
        for i, n_i in enumerate(n):
            n_seq = N[i]
            inv_log_probs_i = -log_probs[i, :n_seq]
            if use_topk:
                loc = inv_log_probs_i.topk(n_i, dim=0).indices + offset
            else:
                loc = (
                    torch.multinomial(torch.softmax(inv_log_probs_i, dim=-1), n_i)
                    + offset
                )
            loc_mask[i].scatter_(dim=-1, index=loc, value=True)

        return loc_mask

    def sample_txt_tok(
        self,
        tok,
        pos,
        loc_img,
        loc_txt,
        loc_spc,
        N,
        return_tokens=False,
        input_loc_mask=None,
    ):
        n_iter = self.cfg.stage2.mask_hparams.i2t_n_steps
        sample_method = self.cfg.sampling.txt_sample_method
        use_argmax_sample = "argmax" in sample_method
        if self.cfg.sampling.txt_mask_sample_method == "topk":
            use_topk = True
        else:
            use_topk = False
        temperature = self.cfg.sampling.txt_temperature
        top_k = self.cfg.sampling.txt_top_k
        top_p = self.cfg.sampling.txt_top_p
        B = tok.shape[0]
        L = loc_img.shape[1]
        T = loc_txt[0].sum()

        offset = loc_txt[0].nonzero()[0]
        attn_mask = self._get_i2t_attn_mask(B, L, T, offset, N, loc_txt.device)

        txt_tok = tok
        prev_log_probs = torch.zeros([B, T], device=loc_txt.device)

        conds = []
        gens = []
        temps = [temperature] * n_iter
        for t in range(n_iter):
            n = N * (n_iter - t) / n_iter
            n = n.int().clip(min=1).tolist()
            loc_mask = self._get_txt_mask(
                loc_txt, prev_log_probs, N, n, use_topk=use_topk
            )

            if input_loc_mask is not None:  # for it2t infilling,
                loc_mask = torch.logical_and(loc_mask, input_loc_mask)

            tok_masked, _loc_spc = self._apply_mask(
                txt_tok, loc_spc, loc_mask, loc_img, loc_txt
            )
            x = self.model(
                tok_masked, pos, loc_img, loc_txt, _loc_spc, attn_mask=attn_mask
            )
            if not hasattr(self.model, "lm_layer"):
                logit_txt = self.model.head_txt(x[loc_txt])
            else:
                logits = self.model.lm_layer(x, self.model.tok_emb)
                logit_txt = logits[:, :, self.model.vocab_size_img :]

            _dist = torch.distributions.categorical.Categorical(logits=logit_txt)
            logit_txt = top_k_top_p_filtering(
                logit_txt / temps[t], top_k=top_k, top_p=top_p
            )
            dist = torch.distributions.categorical.Categorical(logits=logit_txt)
            if use_argmax_sample:
                sampled_txt = torch.argmax(dist.probs, dim=-1)
            else:
                sampled_txt = dist.sample()

            txt_tok = torch.where(loc_mask, sampled_txt, txt_tok)
            conds.append(token2txt(tok_masked[:, -T:], self.tokenizer)[0])
            gens.append(token2txt(txt_tok[:, -T:], self.tokenizer)[0])
            cur_log_probs = dist.log_prob(sampled_txt) * loc_mask
            prev_log_probs = cur_log_probs[:, -T:] + prev_log_probs * torch.logical_not(
                loc_mask[:, -T:]
            )

        txt_tok = txt_tok[:, -T:]
        if return_tokens:
            return txt_tok, prev_log_probs.sum(-1)
        else:
            return token2txt(txt_tok, self.tokenizer)

    def sample_t2i(
        self,
        txt,
        txt_mask,
        ctx_len_img=256,
        n_steps=8,
        strategy="multinomial-maskgit",
        temp_st=1.4,
        temp_end=0.6,
        multi_temp_st=2.0,
        multi_temp_end=0.2,
    ):
        B = txt.shape[0]
        device = txt.device
        tok_img_rep = torch.zeros([B, ctx_len_img]).to(device).int()

        temps = np.linspace(temp_st, temp_end, n_steps)
        multi_temps = np.linspace(multi_temp_st, multi_temp_end, n_steps)
        rs = np.linspace(0, 1, n_steps + 1)[1:]

        with torch.no_grad():
            tok, pos, loc_img, loc_txt, loc_spc = self._get_input(
                tok_img_rep, txt, task="t2i"
            )
            loc_spc_wo_mask = loc_spc.clone()
            L = tok_img_rep.shape[1]
            loc_mask, attn_mask, r = self._get_mask(
                loc_img, loc_txt, loc_spc, r=0.0, task="t2i", txt_mask=txt_mask
            )
            for step in range(n_steps):
                tok_masked, loc_spc = self._apply_mask(
                    tok, loc_spc_wo_mask, loc_mask, loc_img, loc_txt
                )
                x = self.model(
                    tok_masked, pos, loc_img, loc_txt, loc_spc, attn_mask=attn_mask
                )

                with autocast(enabled=True):
                    logit_txt, logit_img, tok_txt = self.get_logits_and_target_pair(
                        x, txt, loc_img, loc_txt
                    )
                    logit_txt /= temps[step]
                    logit_img /= temps[step]

                prob_txt = F.softmax(logit_txt, dim=-1)
                prob_img = F.softmax(logit_img, dim=-1)

                sampled_txt = torch.distributions.categorical.Categorical(
                    logits=logit_txt
                ).sample()
                sampled_img = torch.distributions.categorical.Categorical(
                    logits=logit_img
                ).sample()

                prob = torch.ones(x.shape[:-1], dtype=prob_txt.dtype).to(x.device)
                prob[loc_txt] = torch.take_along_dim(
                    prob_txt, torch.unsqueeze(sampled_txt, -1), -1
                ).squeeze(-1)
                prob[loc_img] = torch.take_along_dim(
                    prob_img, torch.unsqueeze(sampled_img, -1), -1
                ).squeeze(-1)

                prev_loc_mask = loc_mask.clone()
                loc_mask = self._get_next_mask(
                    prev_loc_mask,
                    prob,
                    rs[step],
                    L,
                    strategy,
                    multi_temps[step],
                    is_last_step=step == n_steps - 1,
                )
                gen_mask = torch.logical_and(prev_loc_mask, torch.logical_not(loc_mask))
                tok[gen_mask] = sampled_img[gen_mask[loc_img]]

        pixels = sampled_img * loc_mask[loc_img] + tok[loc_img] * torch.logical_not(
            loc_mask[loc_img]
        )
        pixels = pixels.reshape([x.shape[0], -1])
        hh = int(math.sqrt(pixels.shape[-1]))
        pixels = pixels.reshape(pixels.shape[0], hh, hh)
        pixels = self.model_vqvae.decode_code(pixels) * 0.5 + 0.5
        pixels = torch.clamp(pixels, 0, 1) * 255
        pixels = pixels.cpu().to(torch.uint8).permute(0, 2, 3, 1)
        pixels = torch.split(pixels, 1)
        pixels = [Image.fromarray(pixel.squeeze().numpy()) for pixel in pixels]

        return pixels

    def sample_it2i(
        self,
        txt,
        txt_mask,
        ctx_len_img=256,
        n_steps=8,
        strategy="multinomial-maskgit",
        temp_st=1.4,
        temp_end=0.6,
        multi_temp_st=2.0,
        multi_temp_end=0.2,
        source_img=None,
        img_mask=None,
    ):

        B = txt.shape[0]
        tok_img_rep = self.model_vqvae.get_codes(source_img).detach()
        img_mask_rep = img_mask.bool().squeeze(1)
        if source_img.shape[0] == 1:
            tok_img_rep = torch.repeat_interleave(tok_img_rep, B, dim=0)
            img_mask_rep = torch.repeat_interleave(img_mask_rep, B, dim=0)
        img_mask_rep = img_mask_rep.reshape([B, -1])

        temps = np.linspace(temp_st, temp_end, n_steps)
        multi_temps = np.linspace(multi_temp_st, multi_temp_end, n_steps)
        rs = np.linspace(0, 1, n_steps + 1)[1:]

        with torch.no_grad():
            tok, pos, loc_img, loc_txt, loc_spc = self._get_input(
                tok_img_rep, txt, task="t2i"
            )
            loc_spc_wo_mask = loc_spc.clone()
            offset = loc_img[0].nonzero()[0]

            L = tok_img_rep.shape[1] - img_mask_rep.sum(dim=-1).cpu()

            loc_mask, attn_mask, r = self._get_mask(
                loc_img, loc_txt, loc_spc, r=0.0, task="t2i", txt_mask=txt_mask
            )
            loc_img_mask = torch.zeros_like(loc_mask).bool()
            loc_img_mask[:, offset:][img_mask_rep] = True
            loc_mask = torch.logical_and(loc_mask, ~loc_img_mask)
            for step in range(n_steps):
                tok_masked, loc_spc = self._apply_mask(
                    tok, loc_spc_wo_mask, loc_mask, loc_img, loc_txt
                )
                x = self.model(
                    tok_masked, pos, loc_img, loc_txt, loc_spc, attn_mask=attn_mask
                )

                with autocast(enabled=True):
                    logit_txt, logit_img, tok_txt = self.get_logits_and_target_pair(
                        x, txt, loc_img, loc_txt
                    )
                    logit_txt /= temps[step]
                    logit_img /= temps[step]

                prob_txt = F.softmax(logit_txt, dim=-1)
                prob_img = F.softmax(logit_img, dim=-1)

                sampled_txt = torch.distributions.categorical.Categorical(
                    logits=logit_txt
                ).sample()
                sampled_img = torch.distributions.categorical.Categorical(
                    logits=logit_img
                ).sample()

                prob = torch.ones(x.shape[:-1], dtype=prob_txt.dtype).to(x.device)
                prob[loc_txt] = torch.take_along_dim(
                    prob_txt, torch.unsqueeze(sampled_txt, -1), -1
                ).squeeze(-1)
                prob[loc_img] = torch.take_along_dim(
                    prob_img, torch.unsqueeze(sampled_img, -1), -1
                ).squeeze(-1)

                prev_loc_mask = loc_mask.clone()
                loc_mask = self._get_next_mask(
                    prev_loc_mask,
                    prob,
                    rs[step],
                    L,
                    strategy,
                    multi_temps[step],
                    is_last_step=step == n_steps - 1,
                )
                gen_mask = torch.logical_and(prev_loc_mask, torch.logical_not(loc_mask))
                tok[gen_mask] = sampled_img[gen_mask[loc_img]]

        pixels = sampled_img * loc_mask[loc_img] + tok[loc_img] * torch.logical_not(
            loc_mask[loc_img]
        )
        pixels = pixels.reshape([x.shape[0], -1])
        hh = int(math.sqrt(pixels.shape[-1]))
        pixels = pixels.reshape(pixels.shape[0], hh, hh)
        pixels = self.model_vqvae.decode_code(pixels) * 0.5 + 0.5
        pixels = torch.clamp(pixels, 0, 1) * 255
        pixels = pixels.cpu().to(torch.uint8).permute(0, 2, 3, 1)
        pixels = torch.split(pixels, 1)
        pixels = [Image.fromarray(pixel.squeeze().numpy()) for pixel in pixels]

        return pixels

    def sample_i2t(self, batch: Items, tok_img=None, sample_size=1, return_tokens=False, input_text=False):
        with torch.no_grad():
            if tok_img is None:
                tok_img = self.model_vqvae.get_codes(batch.img).detach()

            expanded_return_idx = (
                torch.arange(tok_img.shape[0]).view(-1, 1).repeat(1, sample_size).view(-1).to(tok_img.device)
            )
            if not input_text: # i2t case
                inputs = torch.ones_like(batch.txt) * self.tokenizer.eos_token_id
            else: # it2t, text-infilling case
                inputs = batch.txt
            tok, pos, loc_img, loc_txt, loc_spc = self._get_input(tok_img, inputs, task='i2t')
            loc_mask, attn_mask, r = self._get_mask(loc_img, loc_txt, loc_spc, r=1, task='i2t', txt_mask=batch.txt_mask)

            if input_text:
                txt_mask = (inputs != self.tokenizer.eos_token_id).int()
                len = torch.sum(txt_mask)
                mask_len = len // 2  # 50% center masking
                txt_mask[0, int(len // 2 - mask_len // 2): int(len // 2 - mask_len // 2) + mask_len] = 0
                txt_mask_pad = (inputs == self.tokenizer.eos_token_id).int()
                txt_mask = (txt_mask + txt_mask_pad).flatten()

                B = tok_img.shape[0]
                txt_mask_rep = txt_mask.bool()
                if inputs.shape[0] == 1:
                    txt_mask_rep = torch.repeat_interleave(txt_mask_rep, B, dim=0)
                txt_mask_rep = txt_mask_rep.reshape([B, -1])
                offset = loc_txt[0].nonzero()[0]
                loc_txt_mask = torch.zeros_like(loc_mask).bool()
                loc_txt_mask[:, offset:][txt_mask_rep] = True
                loc_mask = torch.logical_and(loc_mask, ~loc_txt_mask)

            tok_masked, _loc_spc = self._apply_mask(tok, loc_spc, loc_mask, loc_img, loc_txt)
            x = self.model(tok_masked, pos, loc_img, loc_txt, _loc_spc, attn_mask=attn_mask)
            length_logits = self.predict_length(x)
            dist = torch.distributions.categorical.Categorical(logits=length_logits)
            N = torch.argmax(dist.probs, dim=-1) + 1

            if sample_size > 1:
                tok = tok.index_select(0, expanded_return_idx)
                pos = pos.index_select(0, expanded_return_idx)
                loc_img = loc_img.index_select(0, expanded_return_idx)
                loc_txt = loc_txt.index_select(0, expanded_return_idx)
                loc_spc = loc_spc.index_select(0, expanded_return_idx)
                N = N.index_select(0, expanded_return_idx)
                n_var_high = 1
                length_variation = torch.randint(n_var_high, N.size(), device=N.device)
                length_variation = torch.where(torch.cuda.FloatTensor(N.size()).uniform_() > 0.5, -length_variation, length_variation)
                N += length_variation
            if input_text:
                return self.sample_txt_tok(tok, pos, loc_img, loc_txt, loc_spc, N, return_tokens=return_tokens,
                                           input_loc_mask=loc_mask)
            else:
                return self.sample_txt_tok(tok, pos, loc_img, loc_txt, loc_spc, N, return_tokens=return_tokens)


