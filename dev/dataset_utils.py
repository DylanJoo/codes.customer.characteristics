import numpy as np
import pandas as pd
from sklearn import preprocessing
from datasets import Dataset
import unicodedata
import torch
import os
import random

# 0) Helper function
def normalize_string(string, sub='[UNK]'):
    try:
        return unicodedata.normalize("NFKD", string).strip()
    except:
        return sub

# D) Extract per-user purchasing sequence
def get_user_aggregate_df(path):
    # aggreagert user purchased history
    def agg_purchase_items(x):
        d = {}
        purchased_items = np.array(x['item_name'])
        items_seq, items_counts = np.unique(purchased_items, return_counts=True)
        d['purchase_seq'] = items_seq.tolist()
        d['purchase_count'] = items_counts.tolist()
        return pd.Series(d, index=['purchase_seq', 'purchase_count'])

    df = pd.read_csv(path)
    df = df.loc[:, ['user_id', 'item_name', 'item_tag', 'item_brand_name', 'store_brand_name']]
    df = df[~ (df.item_tag.isna() & df.item_brand_name.isna())]
    df = df[~ (df.item_tag.isna() & df.item_brand_name == '無法辨識_Unrecognizable')]
    df_agg = df.groupby(['user_id']).apply(func=agg_purchase_items).reset_index()
    return df_agg

# 3) get hf dataset
def get_dataset(df):

    dataset = Dataset.from_pandas(df)

    def preprocessing(examples):
        n_users = len(examples['purchase_count'])
        for i in range(n_users):
            n_items = len(examples['purchase_seq'][i])
            for j in range(n_items):
                examples['purchase_seq'][i][j] = normalize_string(
                        examples['purchase_seq'][i][j], '[MASK]'
                )

        return examples

    dataset = dataset.map(
            function=preprocessing,
            batched=True
    )
    return dataset


# 4) Build datacollator (before dataloader)
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple, Any
from transformers.tokenization_utils_base import (
        PaddingStrategy,
        PreTrainedTokenizerBase
)
@dataclass
class EInvoicePurchaseSeqDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    truncation: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str] = True
    is_train: Union[bool] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        """
        # setting 0: 
        ------
        input: [CLS] <item_name> | <item_name> | .... 
        output: use [CLS] vecotr (NA) 
        """
        # step1: Random draw
        def random_purchasing_sampling(seq_items, seq_counts, max_length=50):
            ## exceed max length then selected by ranomd sample
            if len(seq_counts) > max_length:
                idx_selected = random.choices(np.arange(len(seq_items)), max_length)
                seq_items = np.take(seq_items, idx_selected)
                seq_counts = np.take(seq_counts, idx_selected)

            idx_sorted = np.argsort(seq_counts)[::-1]
            seq_items = np.take(seq_items, idx_sorted)
            item_name_seq = "|".join(seq_items)

            return item_name_seq

        batch = self.tokenizer(
                [random_purchasing_sampling(ft['purchase_seq'], ft['purchase_count']) \
                        for ft in features],
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
        )

        # labels 
        # [TODO] Build the constrastive learning pair 
        if is_train:
            cate_name_label = [ft['item_tag_lbl'] for ft in features]
            batch['labels'] = torch.tensor(cate_name_label).to(dtype=torch.long)

        return batch

# @dataclass
# class EInvoiceDataCollator:
#     tokenizer: PreTrainedTokenizerBase
#     padding: Union[bool, str, PaddingStrategy] = True
#     truncation: Union[bool, str] = True
#     max_length: Optional[int] = None
#     pad_to_multiple_of: Optional[int] = None
#     return_tensors: str = "pt"
#     padding: Union[bool, str] = True
#     is_train: Union[bool] = True
#
#     def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
#
#         """
#         # setting 0: 
#         ------
#         input: [CLS] <item_name> [SEP] 
#         output: labels [int] (encoded item tag labels)
#         """
#         item_name = [ft['item_name'] for ft in features]
#
#         batch = self.tokenizer(
#             item_name,
#             padding=self.padding,
#             truncation=self.truncation,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors=self.return_tensors,
#         )
#
#         """
#         # setting 1: 
#         ------
#         input: [CLS] <item_name> [SEP] <item_brand_name> <store_brand_name> [SEP]
#         output: labels [int] (encoded item tag labels)
#         example:
#
#         # item_desc = [
#         #         f"{ft['item_brand_name']} {ft['store_brand_name']}" for ft in features
#         # ]
#         """
#
#         # labels
#         if is_train:
#             cate_name_label = [ft['item_tag_lbl'] for ft in features]
#             batch['labels'] = torch.tensor(cate_name_label).to(dtype=torch.long)
#
#         return batch
