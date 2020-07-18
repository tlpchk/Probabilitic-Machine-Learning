"""Utility functions for Pyro."""
from collections import defaultdict
import matplotlib.pyplot as plt
import pyro
import torch
from tqdm.auto import tqdm


def train_svi(model, guide, X, y, num_epochs=500, lr=1e-2):
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

    plt.figure()
    plt.plot(range(num_epochs), losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')

    for pname, pvals in params.items():
        if pname == 'c_logits':
            plt.figure()
            for c in range(3):
                plt.plot(range(num_epochs), [v[c] for v in pvals], label=f'C={c}')
            plt.title('Class logits')
            plt.legend()
        elif any(w in pname for w in ('_mu', '_sigma')):
            plt.figure()
            for c in range(pvals[0].shape[0]):
                vals = [v[c] for v in pvals]
                plt.plot(range(num_epochs), vals, label=f'{pname} | c={c}')
            plt.legend()
            plt.title(pname)
        else:
            fig, axs = plt.subplots(ncols=pvals[0].shape[1], figsize=(15, 5))
            for idx, ax in enumerate(axs):
                for c in range(pvals[0].shape[0]):
                    vals = [v[c][idx] for v in pvals]
                    ax.plot(range(num_epochs), vals, label=f'{pname}_(CAT={idx}) | c={c}')
                ax.legend()
            fig.suptitle(pname)
