import abc
from typing import List

import numpy as np
import torch
import torch.distributions as dists
import tqdm.auto as tqdm
from gensim.corpora.dictionary import Dictionary


class BaseLDA(abc.ABC):
    def __init__(
        self,
        num_optim_steps: int,
        num_topics: int,
        dictionary: Dictionary,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        self.num_topics = num_topics
        self.vocabulary_size = len(dictionary.values())
        self.dictionary = dictionary
        self.num_optim_steps = num_optim_steps
        self.alpha = alpha
        self.beta = beta

        self.topic_assignments = (
            None
        )  # 'z' in the reference, fill in the `find_topic_assignments`
        self.document_topic_distribution = None
        self.word_topics_distribution = None
        self.document_mapping = {}

    @abc.abstractmethod
    def get_perplexity(self, data: List[List[str]], iterations: int) -> float:
        ...

    @abc.abstractmethod
    def run_gibbs_step(self, data: List[List[str]]) -> float:
        ...

    @abc.abstractmethod
    def get_word_probas_over_topics_for_doc(
        self, doc: List[str]
    ) -> np.ndarray:
        ...

    @abc.abstractmethod
    def assign_topics_to_words(
        self,
        data: List[torch.Tensor],
        document_topic_dist_theta: torch.Tensor,
        topic_word_dist_phi: torch.Tensor,
    ) -> List[torch.Tensor]:
        ...

    @abc.abstractmethod
    def estimate_document_topic_distribution_theta(
        self, data: List[torch.Tensor], topic_assignments_z: List[torch.Tensor]
    ) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def estimate_topic_word_distribution_phi(
        self, data: List[torch.Tensor], topic_assignments_z: List[torch.Tensor]
    ) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def get_topic_for_the_document(self, doc: List[str]) -> int:
        ...

    def find_params(self, data: List[List[str]]) -> List[float]:
        # phi
        self.word_topics_distribution = dists.Dirichlet(
            torch.ones(self.num_topics, self.vocabulary_size)
        ).sample()

        # theta
        self.document_topic_distribution = dists.Dirichlet(
            torch.ones(len(data), self.num_topics)
        ).sample()

        # z
        self.topic_assignments = [
            dists.Categorical(
                probas[None].expand([len(data[i]), self.num_topics])
            )
            for i, probas in enumerate(self.document_topic_distribution)
        ]

        history = []
        for index, document in enumerate(data):
            self.document_mapping[" ".join(document)] = index

        for _ in tqdm.trange(self.num_optim_steps, desc="Optim step"):
            self.run_gibbs_step(data)
            history.append(self.get_perplexity(data, -1))
        return history

    def document2index(self, doc: List[str]) -> int:
        """Get index of the document from the dataset.

        :param doc: List of strings containing words.
        :return: Index of the document from the dataset.
        """
        return self.document_mapping[" ".join(doc)]

    def get_word_indices_from_strings(
        self, dataset: List[List[str]]
    ) -> List[torch.Tensor]:
        """Convert each string to the corresponding index in the dictionary.

        :param dataset: Each entry is a lista of strings (a single document)
            with the variable length and the whole parameter represents list of
            documents. In a single document, words are strings. 
        :return: List of tensors. Each tensor is of length of the corresponding
            document and contains indices of words in the dictionary.
        """
        doc_word_indices = [
            torch.tensor(
                [self.dictionary.token2id[token] for token in doc]
            ).long()
            for doc in dataset
        ]
        return doc_word_indices

    @property
    def theta(self) -> torch.Tensor:
        return self.document_topic_distribution

    @property
    def phi(self) -> torch.Tensor:
        return self.word_topics_distribution
