import torch
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Any, List


@dataclass(init=True, repr=True)
class Item:
    imgpath: str = field(default=None)
    img: torch.Tensor = field(default=None, repr=False)
    v_half_img: torch.Tensor = field(default=None, repr=False)
    h_half_img: torch.Tensor = field(default=None, repr=False)
    txt: torch.Tensor = field(default=None)
    txt_mask: torch.Tensor = field(default=None)
    txt_target_mask: torch.Tensor = field(default=None)
    txt_pos_id: torch.Tensor = field(default=None)
    gt_txt: Any = field(default=None)
    cond_txt: Any = field(default=None)
    id: int = field(default=None)
    domain: str = field(default=None)
    target_label: torch.Tensor = field(default=None)


@dataclass
class Items:
    imgpath: list
    img: torch.Tensor
    v_half_img: torch.Tensor
    h_half_img: torch.Tensor
    txt: torch.Tensor
    txt_mask: torch.Tensor
    txt_target_mask: torch.Tensor
    txt_pos_id: torch.Tensor
    gt_txt: list
    cond_txt: list
    id: list
    domain: list
    target_label: torch.Tensor

    def __init__(self, elems: List[Item]):
        data = {key: [] for key in asdict(elems[0]).keys()}
        for elem in elems:
            elem = asdict(elem)
            for key, val in elem.items():
                data[key].append(val)

        # TODO: bind below member variables automatically
        self.imgpath = data['imgpath']
        self.img = torch.stack(data['img'])
        self.v_half_img = data['v_half_img']
        if data['v_half_img'][0] is not None:
            self.v_half_img = torch.stack(data['v_half_img'])
        self.h_half_img = data['h_half_img']
        if data['h_half_img'][0] is not None:
            self.h_half_img = torch.stack(data['h_half_img'])
        self.txt = torch.stack(data['txt'])
        self.txt_mask = torch.stack(data['txt_mask'])
        self.txt_target_mask = data['txt_target_mask']
        if data['txt_target_mask'][0] is not None:
            self.txt_target_mask = torch.stack(data['txt_target_mask'])
        self.txt_pos_id = torch.stack(data['txt_pos_id'])
        self.gt_txt = data['gt_txt']
        self.cond_txt = data['cond_txt']
        self.id = data['id']
        self.domain = data['domain']
        self.target_label = data['target_label']
        if data['target_label'][0] is not None:
            self.target_label = torch.stack(data['target_label'])


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

