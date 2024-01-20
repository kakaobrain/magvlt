import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from magvlt.models.utils import get_positional_enc


@torch.no_grad()
def clip_score(texts, txt_ids, model_ar, model_vqvae, model_clip, preprocess_clip, device):
    pixels = sample_1d_fast_t2i(model_ar.module, txt_ids, temperature=1.0, top_k=256, top_p=None, amp=True, max_seq_len=256)
    pixels = model_vqvae.decode_code(pixels.view(pixels.shape[0], 16, 16)) * 0.5 + 0.5
    pixels = torch.clamp(pixels, 0, 1)

    pixels = pixels.cpu().numpy()
    pixels = np.transpose(pixels, (0, 2, 3, 1))

    images = [preprocess_clip(Image.fromarray((pixel*255).astype(np.uint8))) for pixel in pixels]
    images = torch.stack(images, dim=0).to(device=device)
    texts = clip.tokenize(texts).to(device=device)

    image_features = model_clip.encode_image(images)
    text_features = model_clip.encode_text(texts)

    scores = F.cosine_similarity(image_features, text_features).squeeze()

    return scores


def top_k_logits(logits, k):
    if k is None:
        return logits
    else:
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out


def top_p_probs(probs, p):
    if p is None:
        return probs
    else:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_idx_remove_cond = cum_probs >= p

        sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
        sorted_idx_remove_cond[..., 0] = 0

        indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
        probs = probs.masked_fill(indices_to_remove, 0.0)
        norm_probs = probs / torch.sum(probs, dim=-1, keepdim=True)
        return norm_probs


def inf_nan_checker(logits):
    if torch.sum(torch.isnan(logits)):
        print('WARNING... NaN observed')
        logits[torch.isnan(logits)] = -float('Inf')
    if torch.sum(torch.isinf(logits)):
        print('WARNING... INF observed')
        logits[torch.isnan(logits)] = -float('Inf')
    return logits


@torch.no_grad()
def sample(model, img, txt, temperature=1.0, top_k=None, top_p=None, is_tqdm=True, amp=True):
    B, H, W = img.shape

    model.eval()
    pbar = tqdm(range(H), total=H) if is_tqdm else range(H)

    txt_pos = get_positional_enc(txt, mode='1d')
    img_pos = get_positional_enc(img, mode='2d')

    for h in pbar:
        for w in range(W):
            img_ = img.clone().detach()

            logits_img, logits_txt = model(img_, img_pos, txt, txt_pos, amp=amp)
            logits_img = logits_img.view(B, H, W, -1).to(dtype=torch.float32)

            logits = logits_img[:, h, w, :] / temperature
            logits = inf_nan_checker(logits)
            logits = top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)
            probs = top_p_probs(probs, top_p)

            idx = torch.multinomial(probs, num_samples=1)
            img[:, h, w] = idx[:, 0]

    return img


@torch.no_grad()
def sample_fast(model, img, txt, temperature=1.0, top_k=None, top_p=None, is_tqdm=True, amp=True, max_H=32):
    B, _, W = img.shape

    model.eval()
    pbar = tqdm(range(max_H), total=max_H) if is_tqdm else range(max_H)
    txt_pos = get_positional_enc(txt, mode='1d')
    count = 0

    for h in pbar:
        img_pos = get_positional_enc(img, mode='2d')
        for w in range(W):
            img_ = img.clone().detach()

            logits_img, logits_txt = model(img_, img_pos, txt, txt_pos, amp=amp, sampling=True, sampling_idx=count)
            logits_img = logits_img[:, count, :].to(dtype=torch.float32)
            logits = logits_img / temperature
            logits = inf_nan_checker(logits)

            logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            probs = top_p_probs(probs, top_p)

            idx = torch.multinomial(probs, num_samples=1)
            img[:, h, w] = idx[:, 0]

            count += 1

        if h < max_H - 1:
            img = torch.cat([img, torch.zeros((B, 1, W), device=img.device, dtype=img.dtype)], axis=1)

    return img


@torch.no_grad()
def sample_1d(model, txt, temperature=1.0, top_k=None, top_p=None, is_tqdm=True, amp=True, max_seq_len=256):
    img = None

    model.eval()
    pbar = tqdm(range(max_seq_len), total=max_seq_len) if is_tqdm else range(max_seq_len)
    txt_pos = get_positional_enc(txt, mode='1d')

    for h in pbar:
        if img is None:
            img_ = None
            img_pos_ = None
        else:
            img_ = img.clone().detach()
            img_pos_ = get_positional_enc(img_, mode='1d')

        logits_img, _ = model.sample(img_, img_pos_, txt, txt_pos, amp=amp)
        logits_img = logits_img.to(dtype=torch.float32)
        logits = logits_img / temperature
        logits = inf_nan_checker(logits)
        logits = top_k_logits(logits, top_k)

        probs = F.softmax(logits, dim=-1)
        probs = top_p_probs(probs, top_p)

        idx = torch.multinomial(probs, num_samples=1).clone().detach()
        img = idx if img is None else torch.cat([img, idx], axis=1)

    return img


@torch.no_grad()
def sample_1d_fast_t2i(model, txt, temperature=1.0, top_k=None, top_p=None, is_tqdm=True, amp=True, max_seq_len=256):
    img = None
    past = None

    model.eval()
    pbar = tqdm(range(max_seq_len), total=max_seq_len) if is_tqdm else range(max_seq_len)
    txt_pos = get_positional_enc(txt, mode='1d')

    for cnt, h in enumerate(pbar):
        if img is None:
            img_ = None
            img_pos_ = None
        else:
            img_ = img.clone().detach()
            img_pos_ = get_positional_enc(img_, mode='1d')
            img_ = img_[:, cnt-1].unsqueeze(-1)
            img_pos_ = img_pos_[:, cnt-1].unsqueeze(-1)

        logits_img, present = model.sample_t2i(img_, img_pos_, txt, txt_pos, amp=amp, past=past)
        logits_img = logits_img.to(dtype=torch.float32)
        logits = logits_img / temperature

        present = torch.stack(present).clone().detach()
        if past is None:
            past = [present]
        else:
            past.append(present)

        logits = inf_nan_checker(logits)
        logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        probs = top_p_probs(probs, top_p)

        idx = torch.multinomial(probs, num_samples=1).clone().detach()
        img = idx if img is None else torch.cat([img, idx], axis=1)

    del past
    return img

@torch.no_grad()
def sample_1d_fast_it2i(model, txt, img, mask, temperature=1.0, top_k=None, top_p=None, is_tqdm=True, amp=True):
    B, N = txt.size()
    past = None
    masked_token_idx = torch.where(mask < 1)[0]
    ctx_token_idx = torch.where(mask > 0)[0]
    masked_seq_len = int(torch.sum(1-mask)) # number of masked tokens
    ctx_seq_len = int(torch.sum(mask)) # number of context tokens

    img_ctx = torch.index_select(img, 1, ctx_token_idx).repeat(B, 1)
    img = img.repeat(B, 1)
    img_orig = img.clone().detach()
    pred = img_ctx
    img_pos = ctx_token_idx.repeat(B, 1)

    model.eval()
    pbar = tqdm(range(masked_seq_len), total=masked_seq_len) if is_tqdm else range(masked_seq_len)

    #pbar = tqdm(range(max_seq_len), total=max_seq_len) if is_tqdm else range(max_seq_len)
    txt_pos = get_positional_enc(txt, mode='1d')

    for cnt, h in enumerate(pbar):
        if cnt == 0: # generate the first masked token
            img_ = img_ctx.clone().detach()
            img_pos_ = img_pos
        else:
            img_ = pred.clone().detach()
            img_ = img_[:, ctx_seq_len + cnt-1].unsqueeze(-1)
            img_pos_ = img_pos[:, ctx_seq_len + cnt-1].unsqueeze(-1)

        logits_img, present = model.sample_t2i(img_, img_pos_, txt, txt_pos, amp=amp, past=past, generation_task='cm')
        logits_img = logits_img.to(dtype=torch.float32)
        logits = logits_img / temperature

        present = torch.stack(present).clone().detach()
        if past is None:
            past = [present]
        else:
            past.append(present)

        logits = inf_nan_checker(logits)
        logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        probs = top_p_probs(probs, top_p)

        idx = torch.multinomial(probs, num_samples=1).clone().detach()
        pred = idx if pred is None else torch.cat([pred, idx], axis=1)

        # infill the predicted image tokens into mask
        img[:, masked_token_idx[cnt]] = idx.squeeze()
        # update image token position
        img_pos = torch.cat([img_pos, masked_token_idx[cnt].repeat(B, 1)], axis=1)
    del past

    print(torch.sum(abs(img_orig - img.clone().detach()) > 0))

    return img

def sample_1d_fast_i2t(model, img, temperature=1.0, top_k=None, top_p=None, is_tqdm=True, amp=True, max_seq_len=64, eos_token_id=None, txt=None):
    past = None

    model.eval()
    #pbar = tqdm(range(max_seq_len), total=max_seq_len) if is_tqdm else range(max_seq_len)
    img_pos = get_positional_enc(img, mode='1d')
    dones = torch.tensor([False for _ in range(len(img))], dtype=torch.bool, device=img.device)

    #for cnt, h in enumerate(pbar):
    for cnt, h in enumerate(range(max_seq_len)):
        if txt is None:
            txt_ = None
            txt_pos_ = None
        else:
            txt_ = txt.clone().detach()
            txt_pos_ = get_positional_enc(txt_, mode='1d')
            txt_ = txt_[:, cnt-1].unsqueeze(-1)
            txt_pos_ = txt_pos_[:, cnt-1].unsqueeze(-1)

        if type(model).__name__ == 'DistributedDataParallel':
            logits_txt, present = model.module.sample_i2t(img, img_pos, txt_, txt_pos_, amp=amp, past=past)
        else:
            logits_txt, present = model.sample_i2t(img, img_pos, txt_, txt_pos_, amp=amp, past=past)
        logits_txt = logits_txt.to(dtype=torch.float32)
        logits = logits_txt / temperature

        present = torch.stack(present).clone().detach()
        if past is None:
            past = [present]
        else:
            past.append(present)

        logits = inf_nan_checker(logits)
        logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        probs = top_p_probs(probs, top_p)

        idx = torch.multinomial(probs, num_samples=1).clone().detach()
        txt = idx if txt is None else torch.cat([txt, idx], axis=1)

        if (eos_token_id is not None):
            is_done = idx.squeeze() == eos_token_id
            dones = torch.logical_or(dones, is_done)

        if dones.all():
            break

    del past
    return txt


@torch.no_grad()
def sample_vis(model, img, txt, temperature=1.0, top_k=None, top_p=None, is_tqdm=True, amp=True, max_H=32):
    B, _, W = img.shape

    model.eval()
    pbar = tqdm(range(max_H), total=max_H) if is_tqdm else range(max_H)

    txt_pos = get_positional_enc(txt, mode='1d')
    probss = []

    for h in pbar:
        img_pos = get_positional_enc(img, mode='2d')
        for w in range(W):
            img_ = img.clone().detach()

            logits_img, logits_txt = model(img_, img_pos, txt, txt_pos, amp=amp, sampling=True)
            logits_img = logits_img.view(B, h+1, W, -1).to(dtype=torch.float32)

            logits = logits_img[:, h, w, :] / temperature
            logits = inf_nan_checker(logits)
            probss.append(F.softmax(logits, dim=-1).unsqueeze(1).cpu().numpy())

            logits = top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            probs = top_p_probs(probs, top_p)

            idx = torch.multinomial(probs, num_samples=1)
            img[:, h, w] = idx[:, 0]

        if h < max_H - 1:
            img = torch.cat([img, torch.zeros((B, 1, W), device=img.device, dtype=img.dtype)], axis=1)

    probss = np.concatenate(probss, axis=1)
    probss = np.reshape(probss, (probss.shape[0], 32, 32, 8192))

    return img, probss


@torch.no_grad()
def sample1d_vis(model, txt, temperature=1.0, top_k=None, top_p=None, is_tqdm=True, amp=True, max_seq=256):
    B, _ = txt.shape

    model.eval()
    pbar = tqdm(range(max_seq), total=max_seq) if is_tqdm else range(max_seq)

    img = None
    past = None
    txt_pos = get_positional_enc(txt, mode='1d')
    probss = []

    for cnt, h in enumerate(pbar):
        if img is None:
            img_ = None
            img_pos_ = None
        else:
            img_ = img.clone().detach()
            img_pos_ = get_positional_enc(img_, mode='1d')
            img_ = img_[:, cnt-1].unsqueeze(-1)
            img_pos_ = img_pos_[:, cnt-1].unsqueeze(-1)

        logits_img, present = model.sample(img_, img_pos_, txt, txt_pos, amp=amp, past=past)
        logits_img = logits_img.to(dtype=torch.float32)
        logits = logits_img / temperature

        present = torch.stack(present).clone().detach()
        if past is None:
            past = [present]
        else:
            past.append(present)

        probss.append(F.softmax(logits, dim=-1).unsqueeze(1).cpu().numpy())
        logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        probs = top_p_probs(probs, top_p)

        idx = torch.multinomial(probs, num_samples=1).clone().detach()
        img = idx if img is None else torch.cat([img, idx], axis=1)

    del past
    probss = np.concatenate(probss, axis=1)
    probss = np.reshape(probss, (probss.shape[0], 256, 16384))
    return img, probss
