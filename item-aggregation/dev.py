import pandas 
import json
from dataset_utils import get_textual_df

# load idf table
item2idf = json.load(open('item2idf.json', 'r'))

FILE='../data/2022.aigo.full.data.sample.csv'
df = get_textual_df(FILE, users=True, inv=True)
df = df.fillna(",")
df = df.head(100)

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

item2tokens = tokenization('item_name')
df.item_name.apply('')
