import os
from typing import Optional
from collections.abc import Callable
import random
import numpy as np
import json

import torch
from PIL import Image
from torchvision.datasets import VisionDataset
from magvlt.datamodules.datasets.coco import CocoCaptions
from magvlt.datamodules.tokenizers import TokenizerUtils
from torch.utils.data import ConcatDataset
from magvlt.datamodules.datasets.dataclass import Item, TextInputItem
from magvlt.datamodules.transforms import HalfTransform


class MapStyleDataset(VisionDataset, TokenizerUtils):
    names = {"cc3m", "cc15m", "coco", "coco2014", "nocaps", "coco2014_t2i"}
    splits = {"train", "val", "test", "test_dev"}
    locs = {"bcloud"}

    def __init__(
        self,
        name: str,
        loc: str,
        split: str,
        transform: Optional[Callable] = None,
        tokenizer_type: Optional[str] = None,
        bpe_pdrop: Optional[float] = None,
        text_ctx: int = 77,
        gt_text: bool = False,
        is_test: bool = False,
        use_half_img: bool = False,
        **ignore_kwargs,
    ):
        assert name in self.names, f"{name} is not in {self.names}"
        assert split in self.splits, f"{split} is not in {self.splits}"
        assert loc in self.locs, f"{loc} is not in {self.locs}"

        if name in ["cc3m", "cc15m"]:
            super().__init__("/data/public/rw/datasets/", transform=transform)
        elif name.startswith("coco"):
            super().__init__('/data/karlo-research_715/karlo-backbone/magvlt/it2it/datasets/coco', transform=transform)
        elif name == "nocaps":
            self.transform = transform

        self.name = name
        self.split = split
        self.gt_text = gt_text

        self.half_transform = None
        if use_half_img and self.transform is not None:
            self.half_transform = HalfTransform(self.transform._resolution)

        self.build_tokenizer(tokenizer_type, text_ctx, lowercase=True, dropout=bpe_pdrop)

        self.items = []
        if name in ["cc3m", "cc15m"]:
            if split == "train":
                if name == "cc3m":
                    list_names = [f"{self.root}/cc3m_resized/train_list.txt"]
                elif name == "cc15m":
                    list_names = [
                        f"{self.root}/cc3m_resized/train_list.txt",
                        f"{self.root}/CC12M/resized/cc12m_with_hash_no_url_only_valid.tsv",
                    ]
            else:
                list_names = [f"{self.root}/cc3m_resized/val_list.txt"]

            for idx, list_name in enumerate(list_names):
                for line in open(list_name, "r").readlines():
                    toks = line.strip().split("\t")
                    assert len(toks) == 2
                    (imgpath, text) = toks
                    if split == "train":
                        if idx == 0:
                            self.items.append(
                                (os.path.join(self.root, "cc3m_resized", imgpath), text)
                            )
                        else:
                            self.items.append(
                                (
                                    os.path.join(
                                        self.root,
                                        "CC12M/resized/images",
                                        f"{imgpath}.jpg",
                                    )
                                    if name == "cc15m"
                                    else os.path.join(self.root, "cc3m_resized", imgpath),
                                    text,
                                )
                            )
                    else:
                        self.items.append(
                            (os.path.join(self.root, "cc3m_resized", imgpath), text)
                        )
        elif name.startswith("coco"):
            if split == 'test':
                self.items = ConcatDataset([
                    CocoCaptions(root=f'{self.root}/images/val2014',
                                 annFile=f'{self.root}/annotations/dataset_coco_test.json'),
                ])
            else:
                data_list = []
                if 't2i' in self.name:
                    data_list.append(CocoCaptions(root=f'{self.root}/images/{split}2014', annFile=f'{self.root}/annotations/captions_{split}2014.json'))
                else:
                    data_list.append(CocoCaptions(root=f'{self.root}/images/{split}2014', annFile=f'{self.root}/annotations/captions_{split}2014.json'))
                if name == 'coco':
                    data_list.append(CocoCaptions(root=f'{self.root}/images/{split}2017',
                                                  annFile=f'{self.root}/annotations/captions_{split}2017.json'))

                self.items = ConcatDataset(data_list)

        self.custom_len =  None

    def set_custom_length(self, l):
        assert len(self.items) >= l
        self.custom_len = l

    def __len__(self):
        if self.custom_len is not None:
            return self.custom_len
        return len(self.items)

    def __getitem__(self, item: int):
        if self.name in ["cc3m", "cc15m"]:
            imgpath, txt = self.items[item]
            gt_txt = txt
            img = Image.open(imgpath)

            txt_item = self.get_input(txt, pre_proc=self.pre_caption)
            oimg = self.transform(img)
            domain = None

        elif self.name.startswith("coco"):
            imgpath, img, gt_txt = self.items[item]

            if len(gt_txt) > 5:
                gt_txt = gt_txt[:5]
            elif len(gt_txt) < 5:
                gt_txt.append(gt_txt[:(5 - len(gt_txt))])

            if self.transform:
                oimg = self.transform(img)

            h_half_img, v_half_img = None, None
            if self.half_transform:
                h_half_img, v_half_img = self.half_transform(img)

            # text = ' '.join(text)  # text is a list of sentences. Concat them.
            if self.split == "train":
                rnd_txt = random.randint(0, len(gt_txt)-1)
                txt = gt_txt[rnd_txt]
            else:
                txt = gt_txt[0]

            txt_item = self.get_input(txt, pre_proc=self.pre_caption)
            domain = None

        elif self.name == 'nocaps':
            instance = self.items[item]

            # load image
            imgpath = instance["imgpath"]
            img = Image.open(imgpath).convert("RGB")
            oimg = self.transform(img)

            # prepare text token
            txt = instance["captions"][0] if self.split == "val" else "null sentence"
            gt_txt = instance["captions"] if self.split == "val" else None
            txt_item = self.get_input(txt, pre_proc=self.pre_caption)
            domain = instance["domain"]



        item = Item(imgpath=imgpath, img=oimg, txt=txt_item.txt, txt_mask=txt_item.txt_mask, txt_pos_id=txt_item.pos_id, gt_txt=gt_txt, domain=domain, v_half_img=v_half_img, h_half_img=h_half_img)
        return item

    def set_epoch(self, epoch):
        self.epoch = epoch

