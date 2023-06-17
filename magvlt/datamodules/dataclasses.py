import torch
from dataclasses import dataclass, field


@dataclass(repr=True)
class TextInputItem:
    txt: torch.LongTensor = field(default=None)  # tokens for text
    txt_mask: torch.BoolTensor = field(default=None)
    target_mask: torch.BoolTensor = field(default=None)
    pos_id: torch.LongTensor = field(default=None)

    def __init__(self, txt_tok, txt_mask, target_mask=None):
        pos_id = [-1] * len(txt_mask)
        pos_idx = 0
        for i, mask_el in enumerate(txt_mask.tolist()):
            if mask_el == 0:
                continue
            else:
                pos_id[i] = pos_idx
                pos_idx += 1

        if not isinstance(txt_tok, torch.LongTensor):
            txt_tok = torch.LongTensor(txt_tok)
        self.txt = txt_tok
        self.txt_mask = torch.BoolTensor(txt_mask)
        self.pos_id = torch.LongTensor(pos_id)

        if target_mask is not None:
            self.target_mask = torch.BoolTensor(target_mask)
