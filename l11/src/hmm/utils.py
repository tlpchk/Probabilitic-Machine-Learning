import numpy as np


def test_hmm(hmm, X):
    return logsumexp([
        hmm.predict_proba(x)
        for x in X
    ]) - np.log(len(X))


def parameter_loss(prev_theta, theta):
    if prev_theta == (None, None, None):
        return 1

    prev_pi, prev_tr, prev_em = prev_theta
    pi, tr, em = theta

    loss = 0

    # Initial dist loss
    for k in prev_pi.keys():
        loss += (prev_pi[k] - pi[k]) ** 2

    # Transition prob. loss
    for k in prev_tr.keys():
        loss += (prev_tr[k] - tr[k]) ** 2

    # Emission prob. loss
    for k in prev_em.keys():
        loss += (prev_em[k] - em[k]) ** 2

    return loss


def logsumexp(x):
    """Compute sum of log-probs using log-sum-exp trick."""
    x = np.array(x)
    x_max = max(x)
    return np.log(np.sum(np.exp(x - x_max))) + x_max
