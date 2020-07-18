"""Functions for loading datasets."""
import pandas as pd
from sklearn import datasets as sk_ds
from sklearn import model_selection as sk_ms
import torch


def load_iris_dataset():
    X, y = sk_ds.load_iris(return_X_y=True)
    X = pd.DataFrame(X, columns=[
        'sepal-length',
        'sepal-width',
        'petal-length',
        'petal-width',
    ])

    X_tr, X_te, y_tr, y_te = sk_ms.train_test_split(X, y, train_size=0.8, stratify=y)
    print('Full', X.shape, y.shape)
    print('Train', X_tr.shape, y_tr.shape)
    print('Test', X_te.shape, y_te.shape)

    return {
        'train': {'X': X_tr, 'y': y_tr},
        'test': {'X': X_te, 'y': y_te},
    }


def load_cmc(N=-1):
    # Source: https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
    df = pd.read_csv('data/cmc.data', names=[
        'age', # numerical
        'w-education', # categorical
        'h-education',  # categorical
        'num-children',  # numerical
        'w-religion',  # binary
        'w-working',  # binary
        'h-occupation',  # categorical
        'sol-index',  # categorical
        'media-exposure',  # binary
        'contraceptive-method-used',  # class
    ])

    cat_cols = [
        'w-education', # categorical
        'h-education',  # categorical
        'h-occupation',  # categorical
        'sol-index',  # categorical
    ]
    bin_cols = [
        'w-religion',  # binary
        'w-working',  # binary
        'media-exposure',  # binary
    ]

    for col in cat_cols:
        df[col] = (df[col] - 1).astype('category')

    for col in bin_cols:
        df[col] = df[col].astype('category')    

    if N != -1:
        df = df.sample(
            n=N,
            weights='contraceptive-method-used',
            random_state=2020,
        )

    X = df[df.columns[:-1]]
    y = df['contraceptive-method-used'].values - 1

    X_tr, X_te, y_tr, y_te = sk_ms.train_test_split(X, y, train_size=0.8, stratify=y)
    print('Full', X.shape, y.shape)
    print('Train', X_tr.shape, y_tr.shape)
    print('Test', X_te.shape, y_te.shape)

    return {
        'train': {'X': X_tr.reset_index(drop=True), 'y': y_tr},
        'test': {'X': X_te.reset_index(drop=True), 'y': y_te},
    }
