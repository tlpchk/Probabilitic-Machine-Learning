from typing import Collection, Dict, List, Optional, Union, Tuple

import numpy as np

from src.hmm import baum_welch as bw
from src.hmm import forward_backward as fw_bw
from src.hmm import viterbi as vit
from src.hmm import train
from src.hmm import utils


class HMM:
    def __init__(
        self, 
        Z: Collection[str], 
        X: Collection[str], 
        init_dist: Optional[Dict[str, float]] = None, 
        transition_probs: Optional[Dict[Tuple[str, str], float]] = None, 
        emission_probs: Optional[Dict[Tuple[str, str], float]] = None,
    ):
        self.Z = Z  # Hidden states space
        self.X = X  # Observations space

        self.pi = init_dist  # P(z_0); z_0 -> p
        self.tr = transition_probs  # P(z_t | z_{t-1}); (z_t_1, z_t) -> p
        self.em = emission_probs  # P (x_t | z_t); (x_t, z_t) -> p

    def fit(
        self, 
        X: List[List[str]], 
        algorithm: str = 'viterbi', 
        eps: float = 1e-4, 
        max_epochs: int = 30, 
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Estimates the model's parameters."""
        algorithms_cls = {
            'viterbi': vit.ViterbiTrainingAlgorithm,
            'baum-welch': bw.BaumWelchAlgorithm,
        }
        
        if algorithm not in algorithms_cls.keys():
            raise KeyError(f'No such algorithm: \"{algorithm}\"')
            
        alg = algorithms_cls[algorithm](
            all_X=self.X, all_Z=self.Z,
            eps=eps, max_epochs=max_epochs, 
            verbose=verbose,
        )
            
        theta_hat, logs = alg.train(
            theta_0=(self.pi, self.tr, self.em),
            X=X, 
        )
        self.pi, self.tr, self.em = theta_hat
        
        return logs

    def predict(
        self, 
        X: List[str], 
        return_scores: bool = False,
    ):
        """Computes the most probable sequence of hidden states."""
        if any(param is None for param in (self.pi, self.tr, self.em)):
            raise RuntimeError('Model must be trained first!')

        max_Z, omega = vit.viterbi_decode(
            theta=(self.pi, self.tr, self.em), 
            X=X,
        )

        if return_scores:
            return max_Z, omega

        return max_Z

    def predict_proba(
        self, 
        X: List[str], 
        return_scores: bool = False,
    ):
        """Calculates the log-probability of observing given states."""
        if any(param is None for param in (self.pi, self.tr, self.em)):
            raise RuntimeError('Model must be trained first!')

        log_p, alpha = fw_bw.score_observation_sequence(
            theta=(self.pi, self.tr, self.em), 
            X=X,
        )
        
        if return_scores:
            return log_p, alpha

        return log_p

    def generate(self, N: int):
        """Generates `N` observations and returns both X and Z."""
        X, Z = [], []

        z_all = sorted(self.Z)

        # Draw initial state
        z_0 = np.random.choice(
            a=z_all,
            p=[self.pi[z] for z in z_all],
        )
        Z.append(z_0)

        # Draw next states based on transition matrix
        z_t_1 = z_0
        for _ in range(N - 1):
            z_t = np.random.choice(
                a=z_all,
                p=[self.tr[(z_t_1, z)] for z in z_all],
            )

            Z.append(z_t)
            z_t_1 = z_t

        # Draw observations
        x_all = sorted(self.X)
        for z_t in Z:
            x_t = np.random.choice(
                a=x_all,
                p=[self.em[(x, z_t)] for x in x_all],
            )
            X.append(x_t)

        return X, Z
