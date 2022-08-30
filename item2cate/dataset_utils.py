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

# 1) Extract textual info
def get_textual_df(path):
    df = pd.read_csv(path)
    df = df.loc[:, ['item_name', 'item_tag', 'item_brand_name', 'store_brand_name']]
    ## remove the item without name and brand name
    df = df[~ (df.item_tag.isna() & df.item_brand_name.isna())]
    ### Beside None, "無法辨識_Unrecognizable" 
    df = df[~ (df.item_tag.isna() & df.item_brand_name == '無法辨識_Unrecognizable')]
    return df

# 2) Get label
def encode_item_tag(df):
    le = preprocessing.LabelEncoder()
    cate_labels = le.fit_transform(df.item_tag.to_numpy())
    df['item_tag_lbl'] = cate_labels

    ## output the mapping text files
    with open("category.mapping.tsv", 'w') as f:
        for i, cate in enumerate(le.classes_):
            f.write(f"{i}\t{cate}\n")

    return df, len(le.classes_)

# 3) get hf dataset
def get_dataset(df):

    dataset = Dataset.from_pandas(df)

    def preprocessing(examples):
        n = len(examples['item_name'])
        for i in range(n):
            examples['item_name'][i] = \
                    normalize_string(examples['item_name'][i], '[MASK]')
            # [ASSUMPTION] item_brand_name is label by human
            examples['item_brand_name'][i] = \
                    normalize_string(examples['item_brand_name'][i], '無法辨識_Unrecognizable')
            examples['store_brand_name'][i] = \
                    normalize_string(examples['store_brand_name'][i], '無店家')
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
class EInvoiceDataCollator:
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
        input: [CLS] <item_name> [SEP] 
        output: labels [int] (encoded item tag labels)
        """
        item_name = [ft['item_name'] for ft in features]

        batch = self.tokenizer(
            item_name,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        """
        # setting 1: 
        ------
        input: [CLS] <item_name> [SEP] <item_brand_name> <store_brand_name> [SEP]
        output: labels [int] (encoded item tag labels)
        """

        # item_desc = [
        #         f"{ft['item_brand_name']} {ft['store_brand_name']}" for ft in features
        # ]

        # labels
        if is_train:
            cate_name_label = [ft['item_tag_lbl'] for ft in features]
            batch['labels'] = torch.tensor(cate_name_label).to(dtype=torch.long)

        return batch
