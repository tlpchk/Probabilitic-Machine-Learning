import numpy as np


def forward(theta, X):
    """Implement the forward algorithm.

    :param theta: Parameters of the HMM. Tuple of 3 elements: initial
        state distribution, transition probability, emission probability.
    :param X: Sequence of elements of length T.
    :return: alpha. Dict of log-probabilities, where each key (t, z) denotes
        that any particular state z is chosen at the step t.
    """
    # Implementation here

    alpha = dict()  # (t, z_j) -> log_p

    pi, tr, em = theta

    for t, x in enumerate(X):
        for z in pi.keys():
            if t == 0:
                alpha[(t, z)] = pi[z] * em[(x, z)]
            else:
                alpha[(t, z)] = 0
                for _z in pi.keys():
                    alpha[(t, z)] += alpha[(t - 1, _z)] * tr[(_z, z)]
                alpha[(t, z)] *= em[x, z]
    alpha = {k: np.log(v) for k, v in alpha.items()}
    return alpha


def score_observation_sequence(theta, X):
    """Computes the probability of observing a given sequence.

    :param theta: Parameters of the HMM. Tuple of 3 elements: initial
        state distribution, transition probability, emission probability.
    :param X: Sequence of elements of length T.
    :return: Tuple of log probability and alpha matrix calculated using
        `forward` step. Log probabilities are estimated as a sum of log
        probs over each possible state in the last step of alpha. Note, that
        these probabilities does not have to sum to one, since they
        describe probability of all paths leading to a prticular state.
    """
    # raise NotImplementedError()

    alpha = forward(theta, X)
    prob = 0
    pi, _, em = theta
    for z in pi.keys():
        prob += np.exp(alpha[(len(X) - 1, z)])
    return np.log(prob), alpha


def backward(theta, X):
    """Implement the backward algorithm.

    Similarily to `forward`, it estimates probabilities of a state being
    chosen at the step `t`, but these probabilities are conditioned on
    states in a sequence {T, T-1, ..., 1, 0}.
    :param theta: Parameters of the HMM. Tuple of 3 elements: initial
        state distribution, transition probability, emission probability.
    :param X: Sequence of elements of length T.
    :return: beta. Dict of log-probabilities, where each key (t, z) denotes
        that any particular state z is chosen at the step t.
    """
    beta = {}  # (t, z_i) -> log_p

    pi, tr, em = theta

    # Implementation here

    for z in pi.keys():
        beta[(len(X) - 1, z)] = np.log(1)

    for t in range(len(X) - 1)[::-1]:
        x = X[t + 1]
        for z in pi.keys():
            beta[(t, z)] = np.log(sum([tr[(z, z_)] * em[(x, z_)] * np.exp(beta[(t + 1, z_)]) for z_ in pi.keys()]))

    return beta
