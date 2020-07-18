from collections import defaultdict

import matplotlib.pyplot as plt
import pyro
import torch
from tqdm.auto import tqdm


def train_nb_svi(model, guide, X, y, num_epochs=500, lr=1e-2):
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
            loss = svi.step(X, y)
            history['losses'].append(loss)

            if epoch in torch.linspace(start=0, end=num_epochs, steps=10).int():
                print(f'Loss = {loss}')

            p = dict(pyro.get_param_store())
            for k, v in p.items():
                history['params'][k].append(v.detach().numpy().copy())

    return history


def train_pgm_svi(model, guide, df, num_epochs=500, lr=1e-2, loss=pyro.infer.Trace_ELBO()):
    pyro.clear_param_store()

    svi = pyro.infer.SVI(
        model=model,
        guide=guide,
        optim=pyro.optim.Adam({'lr': lr}),
        loss=loss,
    )

    history = {
        'losses': [],
        'params': defaultdict(list),
    }

    with tqdm(range(num_epochs)) as pbar:
        for epoch in pbar:
            loss = svi.step(df)
            history['losses'].append(loss)

            if epoch in torch.linspace(start=0, end=num_epochs, steps=10).int():
                print(f'Loss = {loss}')

            p = dict(pyro.get_param_store())
            for k, v in p.items():
                history['params'][k].append(v.detach().numpy().copy())

    return history


def visualize_nb_parameters(history):
    num_epochs = len(history['losses'])
    losses = history['losses']
    params = history['params']

    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')

    del (params['c_logits'])

    nrows, ncols = 7, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 30))
    col = 0
    row = 0
    for pname, pvals in params.items():
        for idx in range(pvals[0].shape[1]):
            c = 1
            vals = [v[c][idx] for v in pvals]
            axs[row, col].plot(range(num_epochs), vals, label=f'P({pname[:-5]}={idx}) | c={c}')
        axs[row, col].legend()
        axs[row, col].set_title(pname)
        col = (col + 1) % ncols
        if col == 0:
            row = (row + 1) % nrows


def count_unique(df, column):
    return len(set(df[column]))


def get_all_combinations(pname, df, DG):
    all_combinations = []
    attrs = list(DG.predecessors(pname))
    attrs.append(pname)

    for attr in attrs:
        if not all_combinations:
            all_combinations = [[x] for x in range(count_unique(df, attr))]
        else:
            all_combinations = [i + [x] for x in range(count_unique(df, attr)) for i in all_combinations]
    return all_combinations


def visualize_pgm_parameters(history, df, DG):
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
        var_name = pname[6:]
        combs = get_all_combinations(var_name, df, DG)
        if pname.startswith('theta'):
            preds = list(DG.predecessors(var_name))
            for comb in combs:
                vals = [vals[(tuple(comb))] for vals in pvals]
                conditions = ''
                for i in range(len(comb) - 1):
                    conditions += f'{preds[i]} = {comb[i]}, '
                if len(conditions) > 0:
                    conditions = f'| {conditions[:-2]}'
                label = f'P({pname} = {comb[-1]} {conditions})'
                if count_unique(df, var_name) > 2 or comb[-1] == 1:
                    plt.plot(xs, vals, label=label)
                    plt.legend()
        else:
            y = [val.reshape(-1) for val in pvals]
            plt.plot(y)

        plt.title(pname)
