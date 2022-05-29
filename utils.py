import pandas as pd
import numpy as np
from tqdm import tqdm
from config import Config
from bisect import bisect
from sklearn.model_selection import GroupShuffleSplit
import os
import re
# import fasttext
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import nltk
nltk.download('wordnet')

stemmer = WordNetLemmatizer()

def preprocess_text(document):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(document))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()
        #return document

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if len(word) > 3]

        preprocessed_text = ' '.join(tokens)
        return preprocessed_text

    
def preprocess_df(df):
    """
    This function is for processing sorce of notebook
    returns preprocessed dataframe
    """
    return [preprocess_text(message) for message in df.source]


def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions

def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max

def read_notebook(path):
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
        .assign(id=path.stem)
        .rename_axis('cell_id')
    )

def get_ranks(base, derived):
    return [base.index(d) for d in derived]

def generate_triplet(df, mode='train'):
    triplets = []
    ids = df.id.unique()
    random_drop = np.random.random(size=10000)>0.9
    count = 0

    for id, df_tmp in tqdm(df.groupby('id')):
        df_tmp_markdown = df_tmp[df_tmp['cell_type']=='markdown']

        df_tmp_code = df_tmp[df_tmp['cell_type']=='code']
        df_tmp_code_rank = df_tmp_code['rank'].values
        df_tmp_code_cell_id = df_tmp_code['cell_id'].values

        for cell_id, rank in df_tmp_markdown[['cell_id', 'rank']].values:
            labels = np.array([(r==(rank+1)) for r in df_tmp_code_rank]).astype('int')

            for cid, label in zip(df_tmp_code_cell_id, labels):
                count += 1
                if label==1:
                    triplets.append( [cell_id, cid, label] )
                # triplets.append( [cid, cell_id, label] )
                elif mode == 'test':
                    triplets.append( [cell_id, cid, label] )
                # triplets.append( [cid, cell_id, label] )
                elif random_drop[count%10000]:
                    triplets.append( [cell_id, cid, label] )
                # triplets.append( [cid, cell_id, label] )
        
    return triplets


def get_data():
    NUM_TRAIN = 1500

    paths_train = list((Config.data_dir / 'train').glob('*.json'))[:NUM_TRAIN]
    notebooks_train = [
        read_notebook(path) for path in tqdm(paths_train, desc='Train NBs')
    ]
    df = (
        pd.concat(notebooks_train)
        .set_index('id', append=True)
        .swaplevel()
        .sort_index(level='id', sort_remaining=False)
    )

    df.source = df.source.apply(preprocess_text)

    
    df_orders = pd.read_csv(
        Config.data_dir / 'train_orders.csv',
        index_col='id',
        squeeze=True,
    ).str.split()

    df_orders_ = df_orders.to_frame().join(
        df.reset_index('cell_id').groupby('id')['cell_id'].apply(list),
        how='right',
    )
    
    ranks = {}

    for id_, cell_order, cell_id in df_orders_.itertuples():
        ranks[id_] = {'cell_id': cell_id, 'rank': get_ranks(cell_order, cell_id)}

    df_ranks = (
        pd.DataFrame
        .from_dict(ranks, orient='index')
        .rename_axis('id')
        .apply(pd.Series.explode)
        .set_index('cell_id', append=True)
    )

    df_ancestors = pd.read_csv(Config.data_dir / 'train_ancestors.csv', index_col='id')
    
    df = df.reset_index().merge(df_ranks, on=["id", "cell_id"]).merge(df_ancestors, on=["id"])
    df["pct_rank"] = df["rank"] / df.groupby("id")["cell_id"].transform("count")

    dict_cellid_source = dict(zip(df['cell_id'].values, df['source'].values))

    NVALID = 0.1  # size of validation set

    splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)

    train_ind, val_ind = next(splitter.split(df, groups=df["ancestor_id"]))

    train_df = df.loc[train_ind].reset_index(drop=True)
    val_df = df.loc[val_ind].reset_index(drop=True)


    triplets = generate_triplet(train_df)  #markdown_id, code id, label
    val_triplets = generate_triplet(val_df, mode = 'test')

    return triplets,val_triplets,dict_cellid_source

if __name__ == "__main__":
    triplets,val_triplets = get_data()
    print(triplets)