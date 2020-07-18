import itertools

import pandas as pd
import scipy.stats as stats
from sklearn import model_selection as sk_ms
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_factorized_dataset(path='./data'):
    dataset = pd.read_csv(f"{path}/agaricus-lepiota.csv").astype('category')
    dataset = dataset.apply(lambda x: pd.factorize(x)[0]).astype('category')
    return dataset.drop(['veil-type', 'stalk-root'], axis=1)


def split_dataset(dataset, train_size=0.7):
    X = dataset.loc[:, dataset.columns != 'class']
    y = dataset['class']

    X_tr, X_te, y_tr, y_te = sk_ms.train_test_split(X, y, train_size=train_size, stratify=y)

    return {
        'train': {'X': X_tr, 'y': y_tr},
        'test': {'X': X_te, 'y': y_te}
    }


def generate_random_dag(nodes, p):
    G = nx.gnp_random_graph(len(nodes), p, directed=True)
    DAG = nx.DiGraph([(nodes[u], nodes[v]) for (u, v) in G.edges() if u < v])
    DAG.add_nodes_from(nodes)
    return DAG


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def get_corr(data):
    correlations = np.array([cramers_v(data[x], data[y]) for (x, y) in itertools.product(data.columns, repeat=2)])
    corr = pd.DataFrame(
        correlations.reshape(len(data.columns), -1),
        columns=data.columns,
        index=data.columns
    )
    corr.style.background_gradient(cmap='coolwarm').set_precision(0)
    return corr


def plot_corr(corr):
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        corr,
        mask=mask,
        square=True,
        ax=ax
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
