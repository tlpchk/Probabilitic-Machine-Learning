from typing import Any, Tuple, Union

import numpy as np
import pandas as pd

import common

RNG = np.random.RandomState(101)


def quantize_number(data: Any, precision: int = 3) -> Any:
    return round(data / precision) * precision


def load_betelgeuse_data_random_split() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Betelgeuse dataset.

    The split is performed randomly across all datapoints. Dataset is
    sanitized so processing it should be quite simple.
    :return: Tuple of two dataframes. The first one corresponds to the
        training dataset and to ther to test set. Each dataframe consists
        of two columns: "JD" (X) and "Magnitude" (y). "JD" stands
        for Julian Date commonly used in the astronomy data and Magnitude
        is a logarithmic scale of perceived brightness of the star. The
        higher, the dimmer the object is.
    """
    data = pd.read_csv(common.BETELGEUSE_DATA_PATH, low_memory=False)
    return _preprocess_betelgeuse_data(data, "random")


def load_betelgeuse_data_sequential_split() -> Tuple[
    pd.DataFrame, pd.DataFrame
]:
    """Load Betelgeuse dataset.

    The split was done so left part of the split of sorted datapoints went
    to the train set, and the right part - to the test set. Dataset is
    sanitized so processing it should be quite simple.
    :return: Tuple of two dataframes. The first one corresponds to the
        training dataset and to ther to test set. Each dataframe consists
        of two columns: "JD" (X) and "Magnitude" (y). "JD" stands
        for Julian Date commonly used in the astronomy data and Magnitude
        is a logarithmic scale of perceived brightness of the star. The
        higher, the dimmer the object is.
    """
    data = pd.read_csv(common.BETELGEUSE_DATA_PATH, low_memory=False)
    return _preprocess_betelgeuse_data(data, "sequential")


def _preprocess_betelgeuse_data(
    frame: pd.DataFrame, split_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    magnitude: pd.Series = frame["Magnitude"]
    non_parseable_magnitudes = []

    for val in magnitude:  # type: Union[str, float]
        try:
            float(val)
        except ValueError:
            # there should 9 elems in total
            non_parseable_magnitudes.append(val)

    frame = frame[~frame["Magnitude"].isin(non_parseable_magnitudes)].copy()

    frame = frame[frame["Band"] == "V"]

    frame.loc[:, "Magnitude"] = frame.loc[:, "Magnitude"].astype("float")
    frame.loc[:, "JD"] = frame["JD"].apply(
        lambda x: quantize_number(round(x, 0), 10)
    )
    frame = frame.sort_values(by="JD")
    frame = frame.iloc[
        int(len(frame) * common.BETELGEUSE_PERCENTAGE_POINTS_TO_OMMIT) :
    ]
    frame = frame.groupby("JD").agg({"Magnitude": "mean"}).reset_index()
    frame.rename({"Magnitude_mean": "Magintude"}, inplace=True)
    frame.columns = ["JD", "Magnitude"]
    frame.loc[:, "JD"] = frame["JD"].astype("float")
    frame.loc[:, "JD"] -= frame["JD"].min()
    frame = frame.iloc[:: common.BETELGEUSE_TAKE_EVERY_NTH_POINT]

    return _split_betelguese_to_train_test(frame, split_type)


def _split_betelguese_to_train_test(
    data: pd.DataFrame, split_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split_type == "random":
        indices = RNG.permutation(len(data))
    elif split_type == "sequential":
        indices = np.arange(len(data))
    else:
        raise ValueError(f"Unknown split type {split_type}")

    split_point = int(len(data) * common.TRAIN_RATIO)

    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)

    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]
    return train_data, test_data
