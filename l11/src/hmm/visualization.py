from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.hmm import utils


def hmm_basic_info(hmm):
    """Displays parameters of given HMM."""
    display(Markdown('## Basic info for HMM'))
    display(_pi_to_df(hmm.pi))
    display(_tr_to_df(hmm.Z, hmm.tr))
    display(_em_to_df(hmm.X, hmm.Z, hmm.em))


def print_table(scores, T, Z, name):
    df = pd.DataFrame(
        columns=[f't = {t}' for t in range(T)],
        index=[f'z = \"{z}\"' for z in Z],
    )

    for (t, z), v in scores.items():
        df[f't = {t}'][f'z = \"{z}\"'] = np.round(v, 4)

    display(Markdown(f'## {name} table'))
    display(df)


def print_viterbi_table(scores, T, Z):
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    df = pd.DataFrame(
        columns=[f't = {t}' for t in range(T)],
        index=[f'z = \"{z}\"' for z in Z],
    )

    for (t, z), v in scores.items():
        df[f't = {t}'][f'z = \"{z}\"'] = (np.round(v[0], 4), v[1])


    df = df.style.apply(highlight_max)

    display(Markdown('## Viterbi table'))
    display(df)


def training_plots(hmm, losses, probas, X_test):
    fig, ax = plt.subplots(ncols=2, figsize=(15, 4))
    ax[0].plot(range(len(losses)), losses, marker='x', linestyle='--')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')

    ax[1].plot(range(len(probas)), probas, marker='x', linestyle='--', label='Train')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Train log-prob')

    logp = utils.test_hmm(hmm=hmm, X=X_test)
    print(f'Test log-p: {logp}')
    ax[1].axhline(logp, linestyle='--', color='r', label='Test')
    ax[1].legend()


def _pi_to_df(pi):
    df = pd.DataFrame(
        columns=[f'z_0 = \"{z_0}\"' for z_0 in pi.keys()],
        index=['P(z_0)'],
    )

    for z_0, p_z_0 in pi.items():
        df[f'z_0 = \"{z_0}\"']['P(z_0)'] = p_z_0

    return df


def _tr_to_df(Z, tr):
    df = pd.DataFrame(
        columns=[f'z_t = \"{dst}\"' for dst in Z],
        index=[f'z_t_1 = \"{src}\"' for src in Z],
    )

    for (src, dst), val in tr.items():
        df[f'z_t = \"{dst}\"'][f'z_t_1 = \"{src}\"'] = val

    return df


def _em_to_df(X, Z, em):
    df = pd.DataFrame(
        columns=[f'z_t = \"{z}\"' for z in Z],
        index=[f'x_t = \"{x}\"' for x in X],
    )

    for (x, z), val in em.items():
        df[f'z_t = \"{z}\"'][f'x_t = \"{x}\"'] = val

    return df

