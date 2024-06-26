# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Datasets for converting to MDS Shards."""
import os
import warnings
from typing import Dict, Iterable, Union

import datasets as hf_datasets
import numpy as np
from torch.utils.data import IterableDataset
import sentencepiece as spm
from sentencepiece import SentencePieceProcessor

LANGS = ['en', 'zh', 'id', 'ms', 'tl', 'my', 'th', 'lo', 'km', 'ta', 'vi', 'python', 'javascript', 'shell', 'sql']
SPECIAL_LANG_TOKENS_IDS = {'<|en|>': 31, '<|zh|>': 32, '<|id|>': 33, '<|ms|>': 34, '<|tl|>': 35, '<|my|>': 36, '<|th|>': 37, '<|lo|>': 38, '<|km|>': 39, '<|ta|>': 40, '<|vi|>': 41, '<|python|>': 42, '<|javascript|>': 43, '<|shell|>': 44, '<|sql|>': 45}

KEYS = ['text','raw_contents','contents','raw_content','content']

class ConcatTokensDataset(IterableDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is non-writeable
        # and torch does not support non-writeable tensors, so you get a scary warning and
        # if you do try to write to the tensor you get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(
        self,
        hf_dataset: Union[hf_datasets.IterableDataset, hf_datasets.Dataset],
        tokenizer: SentencePieceProcessor,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
        use_lang_id: bool
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        # os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.max_length = max_length
        self.bos_text = bos_text
        self.eos_text = eos_text
        self.should_wrap = not no_wrap
        self.use_lang_id = use_lang_id

        self.bos_tokens = []
        self.eos_tokens = [tokenizer.piece_to_id(f'<|endoftext|>')]
        if len(self.eos_tokens) > 1:
            warnings.warn(
                f'You specified --concat_tokens with --eos_text, but your EOS text is not tokenizing to one token\
                , instead we got {self.eos_tokens}. Quit if this was in error.')
        # print(self.eos_tokens)
        assert len(self.eos_tokens) == 1

    def __iter__(self) -> Iterable[Dict[str, bytes]]:

        buffer = []
        for sample in self.hf_dataset:
            for key in KEYS:
                if key in sample:
                    iids = self.tokenizer.encode(sample[key])
                    break
            else:
                warnings.warn(f"Skipping object without any of the keys: {sample}")
            if self.use_lang_id:
                # lang_id_tokens = self.tokenizer.encode(f"<|{sample['lang']}|>")
                # lang_id_tokens = [self.tokenizer.piece_to_id(f"<|{sample['lang']}|>")]
                # lang_id_tokens = [SPECIAL_LANG_TOKENS_IDS[f"<|{sample['lang']}|>"]]

                if 'lang' not in sample:
                    sample['lang'] = sample['id'].split('-')[1] 

                lang_id_tokens = [SPECIAL_LANG_TOKENS_IDS.get(f"<|{sample['lang']}|>", SPECIAL_LANG_TOKENS_IDS[f"<|en|>"])]
            else:
                lang_id_tokens = []
            # print(lang_id_tokens, sample['lang'])

            assert len(lang_id_tokens) == 1 or len(lang_id_tokens) == 0

            buffer = buffer + self.bos_tokens + lang_id_tokens + iids + self.eos_tokens
            while len(buffer) >= self.max_length:
                concat_sample = buffer[:self.max_length]
                buffer = buffer[self.max_length:] if self.should_wrap else []
                yield {
                    # convert to bytes to store in MDS binary format
                    'tokens': np.asarray(concat_sample).tobytes()
                }