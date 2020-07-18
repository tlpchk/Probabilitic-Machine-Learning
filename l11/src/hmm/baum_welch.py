from src.hmm import train


class BaumWelchAlgorithm(train.TrainingAlgorithm):
    
    def estimate_parameters(self, theta, X):
        """Estimate parameters of the HMM usng Baum-Welch algorithm.

        :param theta: Parameters of the HMM. Tuple of 3 elements: initial
            state distribution, transition probability, emission probability.
        :param X: Sequence of elements of length T.
        :return: Tuple of parameters of the HMM: initial state probability pi,
            transition probability matrix a, and emission probabiliy matrix b.
        """
        raise NotImplementedError()
