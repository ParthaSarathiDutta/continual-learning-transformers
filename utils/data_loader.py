import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

def load_dataset_csv(file_path):
    df = pd.read_csv(file_path)
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)

def load_all_tasks():
    imdb_train, imdb_test = load_dataset_csv("data/imdb.csv")
    ag_train, ag_test = load_dataset_csv("data/ag_news.csv")
    snips_train, snips_test = load_dataset_csv("data/snips.csv")
    return [(imdb_train, imdb_test), (ag_train, ag_test), (snips_train, snips_test)]
