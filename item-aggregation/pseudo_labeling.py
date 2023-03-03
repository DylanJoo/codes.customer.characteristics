import os
import re
import string
import json
import numpy as np
import pandas as pd
from dataset_utils import get_textual_df
from ckip_transformers.nlp import CkipWordSegmenter
from multiprocessing import Pool

FILE='../data/2022.aigo.full.data.csv'

################### utils ###################
def normalized(x):
    x = re.sub(f"\s+", ",", x)
    x = re.sub(f"[{'/_'+string.punctuation}]+\ *", ",", x)
    x = re.sub(f"\[", ",", x)
    x = re.sub(f"\]", ",", x)
    x = re.sub(f"【", ",", x)
    x = re.sub(f"】", ",", x)
    return x

def postprocess(xlist):
    xlist =  [x.replace(',', '') for x in xlist]
    return [x for x in xlist if x != '']

def token_selection(x):
    # item2idf
    idfs = []
    for w in x:
        idfs.append(item2idf[w])
    return x
################### utils ###################

# step 0: Preprocessing
df = get_textual_df(FILE, users=True, inv=True)
df = df.fillna(",")

users = df.user_id.values
inv_ids = df.inv_num.values
df['item_name_preprocessed'] = df.item_name.apply(normalized)
# df['item_tag_preprocessed'] = df.item_tag.apply(normalized)
# df['item_brand_name_preprocessed'] = df.item_brand_name.apply(normalized)
# df['store_brand_name_preprocessed'] = df.store_brand_name.apply(normalized)
def cate_mapping():
    if os.path.exists("item2cate.json"):
        mapping = json.load(open('item2cate.json', 'r'))
    else:
        item = df.item_name.tolist()
        cate = df.item_tag.tolist()
        mapping = dict(zip(item, cate))

        with open(f'item2cate.json', 'w') as f:
            json.dump(mapping, f, ensure_ascii=False)
    print(f"{len(mapping)} items with {len(np.unique(mapping.values()))} categories in total.")

item2cate = cate_mapping()

# step 1: Tokenization in batch
def tokenization(col='item_name'):
    if os.path.exists(f'{col}2tokens.json'):
        mapping = json.load(open(f'{col}2tokens.json', 'r'))
    else:
        ws = CkipWordSegmenter(model_name="ckiplab/bert-tiny-chinese-ws")
        words = df[f"{col}_preprocessed"].unique()
        tokens = ws(words, batch_size=1000)
        tokens = [[postprocess(t)] for t in tokens]
        mapping = dict(zip(words, tokens))

        with open(f'{col}2tokens.json', 'w') as f:
            json.dump(mapping, f, ensure_ascii=False)

    return mapping

item2token = tokenization(col='item_name')

# step 2: Merge into primary table
df_tokenizations = pd.DataFrame.from_dict(
        item2token, 
        orient='index', 
        columns=['item_name_tokenized']
)
df = df.merge(df_tokenizations, left_on='item_name_preprocessed', right_index=True)
# df.drop('item_name_preprocessed', axis=1, inplace=True)

# step 3: IDF vectorizer
item_toks_per_user = df.groupby('user_id')['item_name_tokenized'].apply(
        # lambda x: [x1 for x2 in x for x1 in x2]
        lambda x: " ".join([x1 for x2 in x for x1 in x2])
).reset_index()['item_name_tokenized']

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", stop_words=[])
# vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b")
X = vectorizer.fit_transform(item_toks_per_user.to_list())
vectorizer.get_feature_names_out()
item2idf = dict(zip(vectorizer.vocabulary_.keys(), vectorizer.idf_))

with open('item2idf.json', 'w') as f:
    json.dump(item2idf, f, ensure_ascii=False)

# save for checking
df_idf = pd.DataFrame({"word": vectorizer.vocabulary_.keys(), "idf": vectorizer.idf_ })
df_idf.to_csv('item2idf.csv')
print(X.shape)
print(df_idf.sort_values('idf'))

# step 4: Get reduced labels


