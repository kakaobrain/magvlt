from abc import abstractmethod, ABCMeta

import clip
import torch
from torch.nn import functional as F

from magvlt.datamodules.tokenizers import TokenizerUtils


def inf_nan_checker(logits):
    if torch.sum(torch.isnan(logits)):
        print("WARNING... NaN observed")
        logits[torch.isnan(logits)] = -float("Inf")
    if torch.sum(torch.isinf(logits)):
        print("WARNING... INF observed")
        logits[torch.isnan(logits)] = -float("Inf")
    return logits


class SPECIAL_TOKENS(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def get_length(cls):
        pass


class SPECIAL_TOKENS_BASIC:
    BOI, BOT, BOC, BOG, MASK_IMG, MASK_TXT = range(6)

    @classmethod
    def get_length(cls):
        return 6


class SPECIAL_TOKENS_HNH:
    (
        BOI,
        BOT,
        BOC,
        BOG,
        MASK_IMG,
        MASK_TXT,
        BOG_LEFT,
        BOG_RIGHT,
        BOG_TOP,
        BOG_BOTTOM,
    ) = range(10)

    @classmethod
    def get_length(cls):
        return 10


def get_positional_enc(inputs, mode="1d"):
    device = inputs.device
    if mode == "1d":
        B, N = inputs.shape
        xs_pos = torch.arange(N, device=device).repeat((B, 1))
    elif mode == "2d":
        B, H, W = inputs.shape
        xs_pos_h = torch.arange(H, device=device).repeat(B, W, 1).transpose(1, 2)
        xs_pos_w = torch.arange(W, device=device).repeat(B, H, 1)
        xs_pos = (xs_pos_h, xs_pos_w)
    else:
        raise ValueError("%s positional encoding invalid" % mode)

    return xs_pos


def token2txt(output_ids, tokenizer, skip_special_tokens=True):
    eos_token_id = tokenizer.eos_token_id
    outs = []
    for output in output_ids:
        out = output.tolist()
        if eos_token_id in out:
            out = out[: out.index(eos_token_id)]
        outs.append(out)

    if callable(tokenizer):
        outputs = []
        for output in outs:
            outputs.append(
                tokenizer.decode(output, skip_special_tokens=skip_special_tokens)
            )
    else:
        outputs = tokenizer.decode_batch(outs, skip_special_tokens=skip_special_tokens)

    results = []
    for output in outputs:
        results.append(TokenizerUtils.post_caption(output))

    return results


def top_k_top_p_filtering(
    logits, top_k=5, top_p=0.9, filter_value=-float("Inf"), is_probs=False
):
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        if not is_probs:
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        else:
            cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


@torch.no_grad()
def clip_score(texts, images, model_clip, preprocess_clip, device):
    images = [preprocess_clip(image) for image in images]
    images = torch.stack(images, dim=0).to(device=device)
    texts = clip.tokenize(texts).to(device=device)

    image_features = model_clip.encode_image(images)
    text_features = model_clip.encode_text(texts)

    scores = F.cosine_similarity(image_features, text_features).squeeze()
    rank = torch.argsort(scores, descending=True)

    return rank
