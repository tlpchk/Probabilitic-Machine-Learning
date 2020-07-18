import pyro
import torch
from pyro import distributions as dist
from pyro.infer import config_enumerate, TraceEnum_ELBO, Predictive
from torch.distributions import constraints
from sklearn import metrics as sk_mtr
import networkx as nx

from src.utils import train_pgm_svi

import pandas as pd


def all_vars_was_sampled(samples: [], var_dict: []):
    return set(samples.keys()) == set(var_dict.keys())


def variable_can_be_sampled(variable_name, samples, predecessors):
    return variable_name not in list(samples.keys()) \
           and all(elem in list(samples.keys()) for elem in predecessors)


@config_enumerate(default='parallel')
def model(df, graph, observed, n_categories):
    # init var_dict with empty vaalues
    var_dict = {var: None for var in n_categories.keys()}

    # fill var_dict with data
    for var in df.columns:
        var_dict[var] = torch.tensor(df[var].values)

    # init params
    params = {var: pyro.param(
        f'theta_{var}',
        torch.ones(
            [n_categories[pred] for pred in graph.predecessors(var)]
            + [n_categories[var]]

        ),
        constraint=constraints.simplex,
    ) for var in var_dict.keys()}

    with pyro.plate('data', len(df)):
        samples = {}
        while not all_vars_was_sampled(samples, var_dict):
            for var, values in var_dict.items():
                preds = list(graph.predecessors(var))
                if variable_can_be_sampled(var, samples, preds):
                    ids = tuple(samples[p] for p in preds)

                    sample = pyro.sample(
                        var,
                        dist.Categorical(params.get(var)[ids]),
                        obs=values
                    ).long()

                    samples.update({var: sample})


def guide(df, graph, observed, n_categories):
    pass


class PGM:
    def __init__(self, df, graph=nx.DiGraph(), num_epochs=10):
        self.n_categories = {c: len(pd.unique(df[c])) for c in df.columns}
        self.graph = graph
        self.num_epochs = num_epochs
        self.hist = {}

    def fit(self, X, y=None, epochs=0, lr=0.1):
        if epochs < 1:
            epochs = self.num_epochs

        data = X
        if y is not None:
            data['class'] = y.values
        observed = data.columns

        pyro.clear_param_store()

        hist = train_pgm_svi(
            model=lambda df: model(df, self.graph, observed, n_categories=self.n_categories),
            guide=lambda df: guide(df, self.graph, observed, n_categories=self.n_categories),
            df=data,
            num_epochs=epochs,
            lr=lr,
            loss=TraceEnum_ELBO(max_plate_nesting=1)
        )

        return hist

    def predict(self, X):
        data = X
        observed = data.columns
        if 'class' in data.columns:
            observed = observed.drop(['class'])

        served_model = Predictive(
            model=model,
            guide=guide,
            return_sites=('class',),
            num_samples=1)
        predictions = served_model(data, self.graph, observed, self.n_categories)
        return predictions['class'].squeeze(0)

    def print_statistics(self, train_data, test_data):
        print('train')
        print(sk_mtr.classification_report(
            y_true=train_data['class'],
            y_pred=self.predict(train_data))
        )

        print('test')
        print(sk_mtr.classification_report(
            y_true=test_data['class'],
            y_pred=self.predict(test_data))
        )

    def import_pomegranate_model(self, pg_model, nodes):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)

        for origin_id, dependencies in enumerate(pg_model.structure):
            origin = nodes[origin_id]
            for destination_id in dependencies:
                destination = nodes[destination_id]
                graph.add_edge(origin, destination)
        self.graph = graph
