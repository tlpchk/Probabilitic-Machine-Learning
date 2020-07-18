from typing import List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def sentences_chart(
    lda_model, corpus: List[List[str]], start: int = 0, end: int = 11
):
    """Based on: https://bit.ly/3czAY52"""
    end += 2
    corp = corpus[start:end]
    mycolors = [color for name, color in mcolors.XKCD_COLORS.items()]

    fig, axes = plt.subplots(
        end - start, 1, figsize=(20, (end - start) * 0.95), dpi=160
    )
    axes[0].axis("off")
    for i, ax in enumerate(axes):
        if i > 0:
            corp_cur = corp[i - 1]
            topic_document = lda_model.get_topic_for_the_document(corp_cur)
            word_topics = lda_model.get_word_probas_over_topics_for_doc(
                corp_cur
            ).T
            ax.text(
                0.01,
                0.5,
                "Doc " + str(i - 1) + ": ",
                verticalalignment="center",
                fontsize=16,
                color="black",
                transform=ax.transAxes,
                fontweight=700,
            )

            # Draw Rectangle
            ax.add_patch(
                Rectangle(
                    (0.0, 0.05),
                    0.99,
                    0.90,
                    fill=None,
                    alpha=1,
                    color=mycolors[topic_document],
                    linewidth=2,
                )
            )

            word_pos = 0.06
            for j, (word, topics) in enumerate(zip(corp_cur, word_topics)):
                if j < 14:
                    ax.text(
                        word_pos,
                        0.5,
                        word,
                        horizontalalignment="left",
                        verticalalignment="center",
                        fontsize=16,
                        color=mycolors[topics.argmax()],
                        transform=ax.transAxes,
                        fontweight=700,
                    )
                    word_pos += 0.009 * len(
                        word
                    )  # to move the word for the next iter
                    ax.axis("off")
            ax.text(
                word_pos,
                0.5,
                ". . .",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=16,
                color="black",
                transform=ax.transAxes,
            )

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(
        "Sentence Topic Coloring for Documents: "
        + str(start)
        + " to "
        + str(end - 2),
        fontsize=22,
        y=0.95,
        fontweight=700,
    )
    plt.tight_layout()
    plt.show()


def display_topics(lda_model, top_topics: int = -1, num_words: int = 10):
    topic_word_distributions = lda_model.word_topics_distribution.numpy()
    dictionary = lda_model.dictionary
    if top_topics > 0:
        topic_word_distributions = topic_word_distributions[:top_topics]

    word_indices = np.fliplr(topic_word_distributions.argsort(axis=1))[
        :, :num_words
    ]

    num_cols = 5
    num_rows = len(topic_word_distributions) // num_cols + 1
    fig, axes = plt.subplots(
        num_rows, num_rows, figsize=(3 * num_cols, 2 * num_rows), dpi=160
    )
    axes = np.ravel(axes)
    for i, ax in enumerate(axes):
        ax.axis("off")
        ax.text(
            0.5,
            0.9,
            "Topic # " + str(i + 1),
            horizontalalignment="center",
            fontsize=16,
            color="black",
            transform=ax.transAxes,
            fontweight=700,
        )
        if i < len(word_indices):
            for j, word_index in enumerate(word_indices[i]):
                word = dictionary[word_index]
                ax.text(
                    0.5,
                    0.85 - 0.12 * (j + 1),
                    word,
                    horizontalalignment="center",
                    fontsize=12,
                    color="black",
                    transform=ax.transAxes,
                    fontweight=500,
                )
        else:
            ax.remove()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(
        "Top words for each topic", fontsize=22, y=1.05, fontweight=700
    )
    plt.tight_layout()
    plt.show()


def visualize_history(perplexity_history: List[float]):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.plot(perplexity_history, linestyle="--", marker="x")
