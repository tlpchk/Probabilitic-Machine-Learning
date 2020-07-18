"""Utility functions for training PGM."""

from collections import defaultdict

import matplotlib.pyplot as plt
import pyro
from tqdm.auto import tqdm


def train_svi(model, guide, X, num_epochs=500, lr=1e-2):
    pyro.clear_param_store()

    svi = pyro.infer.SVI(
        model=model,
        guide=guide,
        optim=pyro.optim.Adam({'lr': lr}),
        loss=pyro.infer.Trace_ELBO(),
    )

    history = {
        'losses': [],
        'params': defaultdict(list),
    }

    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:
            loss = svi.step(X)
            history['losses'].append(loss)

            if epoch % 50 == 0:
                print(f'Loss = {loss}')

            p = dict(pyro.get_param_store())
            for k, v in p.items():
                history['params'][k].append(v.detach().numpy().copy())

    return history


def visualize_nb_parameters(history):
    num_epochs = len(history['losses'])
    losses = history['losses']
    params = history['params']

    plt.figure(figsize=(15, 5))
    plt.plot(range(num_epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')

    xs = range(num_epochs)

    for pname, pvals in params.items():
        plt.figure(figsize=(15, 5))

        if pname in ('theta_D', 'theta_I'):
            plt.plot(xs, pvals, label=f'P({pname} = 1)')
        elif pname == 'theta_S':
            for i in (0, 1):
                vals = [v[i] for v in pvals]
                plt.plot(xs, vals, label=f'P({pname} = 1 | I = {i})')
        elif pname == 'theta_G':
            for i in (0, 1):
                for d in (0, 1):
                    for g in (0, 1, 2):
                        vals = [v[i][d][g] for v in pvals]
                        plt.plot(xs, vals, label=f'P({pname} = {g} | I = {i}, D = {d})')
        elif pname == 'theta_L':
            for g in (0, 1, 2):
                vals = [v[g] for v in pvals]
                plt.plot(xs, vals, label=f'P({pname} = 1 | G = {g})')

        plt.legend()
        plt.title(pname)
