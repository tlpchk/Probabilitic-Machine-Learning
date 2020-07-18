import numpy as np

from src.hmm import train


def viterbi_decode(theta, X):
    """Implement the Viterbi decoding algorithm.

    :param theta: Parameters of the HMM. Tuple of 3 elements: initial
        state distribution, transition probability, emission probability.
    :param X: Sequence of elements of length T.
    :return: Tuple of: the most probable path of hidden states and the
        omega matrix. The most probability is chosen in a greedy fashion
        by takin the most probable hidden state in each step. Omega describes
        hidden states probabilities at each `t` step.
    """
    # Compute scores (forward pass)
    pi, tr, em = theta

    omega = {}  # (t, z_t) -> max (log_p, z_{t-1})
    init_z = list(pi.keys())[0]
    max_Z = [init_z] * len(X)  # initial most probable hidden states

    for t, x in enumerate(X):
        for z in pi.keys():
            if t == 0:
                omega[(t, z)] = pi[z] * em[(x, z)]
            else:
                omega[(t, z)] = em[x, z] * np.max([omega[(t - 1, _z)] * tr[(_z, z)] for _z in pi.keys()])

            if omega[(t, z)] > omega[(t, max_Z[t])]:
                max_Z[t] = z
    omega = {k: (np.log2(v), v) for k, v in omega.items()}
    return max_Z, omega


class ViterbiTrainingAlgorithm(train.TrainingAlgorithm):

    def __init__(self, all_X, all_Z, eps, max_epochs, verbose, smoothing=1):
        super().__init__(all_X, all_Z, eps, max_epochs, verbose)
        self._s = smoothing

    def estimate_parameters(self, theta, X):
        """Estimate parameters of the HMM usng Viterbi algorithm.

        :param theta: Parameters of the HMM. Tuple of 3 elements: initial
            state distribution, transition probability, emission probability.
        :param X: Sequence of elements of length T.
        :return: Tuple of parameters of the HMM: initial state probability pi,
            transition probability matrix a, and emission probabiliy matrix b.
        """
        pi, tr, em = theta

        pi_n = {k: 0 for k in pi.keys()}
        tr_n = {k: 0 for k in tr.keys()}
        em_n = {k: 0 for k in em.keys()}

        for x in X:
            max_Z, _ = viterbi_decode(theta, x)

            pi_n[max_Z[0]] += 1

            for t in range(1, len(max_Z)):
                tr_n[(max_Z[t - 1], max_Z[t])] += 1
            for t in range(len(max_Z)):
                em_n[(x[t], max_Z[t])] += 1

        S = sum(pi_n.values())
        pi_n = {k: (v + self._s) / (S + self._s * len(pi_n.values())) for k, v in pi_n.items()}

        for z in pi.keys():
            S = sum(tr_n[(z, z_)] for z_ in pi.keys())

            for z_ in pi.keys():
                tr_n[(z, z_)] = (tr_n[(z, z_)] + self._s) / (S + len(pi.keys()) * self._s)

        for z in pi.keys():
            S = 0
            for k in em_n.keys():
                if k[1] == z:
                    S += em_n[k]

            for k in em_n.keys():
                if k[1] == z:
                    em_n[k] = (em_n[k] + self._s) / (S + self._s * len(em_n.keys()) / len(pi_n.keys()))

        theta = (pi_n, tr_n, em_n)
        return theta
