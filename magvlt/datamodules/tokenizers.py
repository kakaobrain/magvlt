import os
import re
from functools import partial

import numpy as np
from tokenizers import CharBPETokenizer

from magvlt.datamodules.dataclasses import TextInputItem

root_dir = os.path.dirname(os.path.abspath(__file__))

TOKENIZERS = {
    "ClipBPE": partial(
        CharBPETokenizer.from_file,
        vocab_filename=os.path.join(root_dir, "clip-vocab.json"),
        merges_filename=os.path.join(root_dir, "clip-merges.txt"),
        unk_token="[UNK]",
    ),
}


def build_tokenizer(tokenizer_type: str, context_length: int, *args, **kwargs):
    tokenizer = TOKENIZERS[tokenizer_type](*args, **kwargs)
    if tokenizer_type.startswith("CharBPE"):
        tokenizer.add_special_tokens(["[PAD]"])
        tokenizer.enable_padding(
            length=context_length, pad_id=tokenizer.token_to_id("[PAD]")
        )
        tokenizer.enable_truncation(max_length=context_length)
    elif tokenizer_type == "GPT2":
        tokenizer.add_special_tokens({"pad_token": "[PAD]", "unk_token": "[UNK]"})
    elif tokenizer_type in ["ClipBPE", "GPT2BPE"]:
        tokenizer.eos_token_id = tokenizer.token_to_id("<|endoftext|>")
        tokenizer.pad_token_id = tokenizer.token_to_id("<|endoftext|>")
        tokenizer.enable_padding(length=context_length, pad_id=tokenizer.pad_token_id)
        tokenizer.enable_truncation(max_length=context_length)
    elif tokenizer_type == "class":
        tokenizer = None
    return tokenizer


class TokenizerUtils:
    def build_tokenizer(
        self, tokenizer_type, text_ctx, lowercase=True, dropout=0.0, sep_token=None
    ):
        self.text_ctx = text_ctx
        self.tokenizer = build_tokenizer(
            tokenizer_type, text_ctx, lowercase=lowercase, dropout=dropout
        )

        self.sep_token = sep_token
        self.sep_token_id = None
        if sep_token is not None:
            ids, n_txt = self.get_token_ids(sep_token)
            sep_id = ids[: ids.index(self.tokenizer.eos_token_id)]
            self.sep_token_id = sep_id[0]

    def get_n_txt(self, ids):
        n_txt = 0
        for _id in ids:
            if _id == self.tokenizer.eos_token_id:
                break
            n_txt += 1
        return n_txt

    def get_token_ids(self, txt, pre_proc=None, add_token_id=None):
        if pre_proc is not None:
            txt = pre_proc(txt)
        if callable(self.tokenizer):
            output = self.tokenizer(txt, padding="max_length", max_length=self.text_ctx)
            ids = output.input_ids
            if len(ids) > self.text_ctx:
                ids = ids[: self.text_ctx]
        else:
            output = self.tokenizer.encode(txt)
            ids = output.ids

        n_txt = self.get_n_txt(ids) + 1  # including <eos>
        if add_token_id:
            # new_ids = ids[:n_txt - 1] + [add_token_id] + ids[n_txt + 1:]
            # new_ids = [add_token_id] + ids
            # new_ids = [add_token_id] + ids[:n_txt - 1] + [add_token_id] + ids[n_txt:]
            new_ids = ids[: n_txt - 1] + [add_token_id, add_token_id] + ids[n_txt:]
            ids = new_ids[: self.text_ctx]
            n_txt += 2

        return ids, n_txt

    def get_input(self, txt, pre_proc=None, add_token_id=None) -> TextInputItem:
        try:
            input, n_txt = self.get_token_ids(
                txt, pre_proc=pre_proc, add_token_id=add_token_id
            )
        except:
            txt = txt.encode("ascii", "ignore").decode()
            input, n_txt = self.get_token_ids(
                txt, pre_proc=pre_proc, add_token_id=add_token_id
            )

        input_mask = np.ones(len(input))
        input_mask[n_txt:] = 0

        item = TextInputItem(input, input_mask)
        return item

    def get_QA_token_ids(self, question, answer=None, max_ques_len=30):
        ques_ids = self.get_token_ids(self.pre_question(question))
        ques_ids = ques_ids[:max_ques_len] + [self.sep_token_id]
        ret = ques_ids
        if answer is not None:
            ans_ids = self.get_token_ids(self.pre_answer(answer))
            ans_ids = ans_ids[: len(ans_ids) - self.max_ques_len]
            ret += ans_ids

        return ret

    def get_vocab_size(self):
        if callable(self.tokenizer):
            return len(self.tokenizer)
        else:
            return self.tokenizer.get_vocab_size()

    @staticmethod
    def pre_caption(caption):
        caption = (
            caption.lower()
            .lstrip(",.!?*#:;~")
            .replace("-", " ")
            .replace("/", " ")
            .replace("<person>", "person")
            .replace("< person >", "person")
        )

        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        return caption

    @staticmethod
    def post_caption(caption):
        caption = TokenizerUtils.pre_caption(caption)
        caption = caption.replace("<", "").replace(">", "")

        # # remove strings after punctuation
        # idx_punct = len(caption)
        # try:
        #     idx_punct = caption.index(".")
        # except:
        #     pass
        # caption = caption[:idx_punct]
        #
        # # deduplication
        # items = caption.split()
        # temp = items[0]
        # new_caption = [items[0]]
        # for item in items[1:]:
        #     if item != temp:
        #         new_caption.append(item)
        #     temp = item
        # caption = " ".join(new_caption)
        #
        # # add punctuation
        # caption += " ."

        return caption

    @staticmethod
    def pre_answer(answer):
        answer = answer.replace("’", "'")
        return answer

    @staticmethod
    def pre_question(question, max_ques_words=30):
        question = (
            re.sub(
                r"([,.'!?\"()*#:;~])",
                "",
                question.lower(),
            )
            .replace("-", " ")
            .replace("/", " ")
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if max_ques_words is not None and len(question_words) > max_ques_words:
            question = " ".join(question_words[:max_ques_words])

        return question
