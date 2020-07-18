import abc

import numpy as np
from tqdm.auto import tqdm

from src.hmm import forward_backward as fw_bw
from src.hmm import utils


class TrainingAlgorithm(abc.ABC):
    
    def __init__(self, all_X, all_Z, eps, max_epochs, verbose):
        self._all_X = all_X
        self._all_Z = all_Z
        self._eps = eps
        self._max_epochs = max_epochs
        self._verbose = verbose

    def train(self, theta_0, X):
        prev_theta = (None, None, None)
        theta = theta_0
    
        logs = {
            'losses': [],
            'probas': [],
        }
    
        for epoch in tqdm(range(self._max_epochs), 
                          desc='Epochs',
                          disable=not self._verbose):
            # Update params
            prev_theta, theta = theta, self.estimate_parameters(theta, X)
    
            # Compute loss
            loss = utils.parameter_loss(prev_theta, theta)
            probas = utils.logsumexp([
                fw_bw.score_observation_sequence(theta=theta, X=x)[0]
                for x in X
            ]) - np.log(len(X))
            
            logs['losses'].append(loss)
            logs['probas'].append(probas)
    
            # Logging
            if self._verbose:
                print(f'Epoch: {epoch} => '
                      f'Loss: {np.round(loss, 5)}, '
                      f'Log-prob: {probas}')
    
            # Stopping condition
            if loss < self._eps:
                break
    
        return theta, logs
    
    @abc.abstractmethod
    def estimate_parameters(self, theta, X):
        pass
