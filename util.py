"""
util.py: contains useful function which are used by multiple files
"""

__author__ = "Abhishek Rajgaria, Ashish Tiwari"

from typing import *
import pandas as pd
from constants import *
from sklearn.model_selection import train_test_split
from character_level_dataset import CharacterLevelDataset

desired_col = ["url", "result"]
RESULT = "result"


def gen_result_col(row: pd.Series) -> int:
    """
    helper method for creating result column in malicious
    dataframe based on the type column
    """
    if row["type"] == "benign":
        return 0
    return 1


def combine_url_dataset() -> pd.DataFrame:
    """
    Combine url_dataset and malicious_phish dataset
    it only adds non benign data from malicious_phish
    dataset to the url_dataset, in-order to increase
    the desired malicious data size and balance the data.
    """
    url_df = pd.read_csv(URL_DATASET_PATH, index_col=0)
    mal_df = pd.read_csv(MALICIOUS_PHISH_DATASET_PATH)

    # add "result" column to the malicious dataset based on "type"
    mal_df[RESULT] = mal_df.apply(gen_result_col, axis=1)
    combined_url_df = pd.concat(
        [url_df[desired_col], mal_df[mal_df[RESULT] == 1][desired_col]], axis=0
    )

    # remove duplicate urls
    combined_url_df = combined_url_df.drop_duplicates()
    return combined_url_df


def get_train_val_test(
    df: pd.DataFrame,
    test_ratio: int = 0.3,
    val_ratio: int = 0.4,
    random_state: int = 42,
):
    """
    Splits the df into train, val and test in the following ratios
    by default 0.7 : 0.12 : 0.18
    """
    train_data, test_data = train_test_split(
        df, test_size=test_ratio, random_state=random_state, shuffle=True
    )
    # Split temp_data into validation and test sets
    # val_data, test_data = train_test_split(
    #     temp_data, test_size=1 - val_ratio, random_state=random_state
    # )

    return train_data, test_data


def get_char_to_idx_and_reverse(df: pd.DataFrame) -> Tuple[dict, dict]:
    """
    Gives a dictionary of char to index for all characters in our URL database
    Also provided the index to char dictionary
    """
    all_url_char = set("".join(df[URL_COLUMN]))
    char_to_idx = {}
    for idx, char in enumerate(all_url_char):
        char_to_idx[char] = idx

    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    return char_to_idx, idx_to_char


def create_dataset(df: pd.DataFrame, char_to_idx: dict) -> CharacterLevelDataset:
    return CharacterLevelDataset(
        df[URL_COLUMN].values, df[RESULT_COLUMN].values, MAX_SEQ_LEN, char_to_idx
    )


def get_vocab_size(char_to_idx: dict) -> int:
    return len(char_to_idx)


def calculate_fpr(labels: list, predictions: list) -> float:
    false_positive = 0
    negative = 0
    for label, pred in zip(labels, predictions):
        false_positive += label == 0 and pred == 1
        negative += label == 0

    print("false positive count", false_positive, "true negative count", negative)
    return false_positive / negative


if __name__ == "__main__":
    cmb_url = combine_url_dataset()
    print((get_char_to_idx_and_reverse(cmb_url)))
