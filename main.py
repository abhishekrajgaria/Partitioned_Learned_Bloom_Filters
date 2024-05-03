"""
main.py: Main file to executing the vanilla, learned and partitioned_learned Bloom filter
"""

import util
import PLBF
import torch
import argparse
import statistics
import numpy as np
import pandas as pd
from constants import *
from torch.utils.data import DataLoader
from vanilla_bloom_filter import VanillaBloomFilter
from learned_bloom_filter import LearnedBloomFilter
from partitioned_learned_bloom_filter import PartitionedLearnedBloomFilter

parser = argparse.ArgumentParser(description="Description of your script")

parser.add_argument("--train", action="store_true", help="If true train model o/w not")

args = parser.parse_args()

target_fprs = [0.005, 0.01, 0.05]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def evaluate_vanilla_bf(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Evaluate the vanilla bloom filter
    for the given target False positive rates
    This function outputs the Actual false positive on the test data
    and total memory use
    """

    max_element = (int)(train_df[RESULT_COLUMN].value_counts()[1])

    actual_fprs = []
    memory_usages = []

    for target_fpr in target_fprs:
        bloom_filter = VanillaBloomFilter(max_element, target_fpr)
        bloom_filter.add_data(train_df)
        actual_fprs.append(bloom_filter.get_fpr_on_test(test_df))
        memory_usages.append(bloom_filter.get_bloom_filter_size())

    return actual_fprs, memory_usages


def evaluate_learned_bf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    toTrain: bool,
    vocab_size: int,
    char_to_idx: dict,
):
    """
    Evaluate the learned bloom filter
    for the given target False positive rates
    This function outputs the Actual false positive on the test data
    and total memory use
    """

    learned_bloom_filter = LearnedBloomFilter(vocab_size, device)
    train_dataset = util.create_dataset(train_df, char_to_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if toTrain:
        learned_bloom_filter.fit(train_dataloader, epoch=1)
        learned_bloom_filter.save_model(MODEL_FILE_PATH)
    else:
        learned_bloom_filter.load_model(MODEL_FILE_PATH)

    actual_fprs = []
    memory_usages = []

    for target_fpr in target_fprs:
        threshold = learned_bloom_filter.get_threshold(target_fpr, train_dataloader)
        print(f"Threshold for target fpr {target_fpr} is {threshold}")
        # labels, predictions = learned_bloom_filter.get_prediction(threshold, train_dataloader)
        false_negative_urls = learned_bloom_filter.get_fn_urls(
            threshold, train_dataloader
        )
        max_element = len(false_negative_urls)
        print(
            "false negative count that needs to be handled by backup bloom filter",
            max_element,
        )
        learned_bloom_filter.bloom_filter = VanillaBloomFilter(
            max_element, target_fpr / 2
        )
        learned_bloom_filter.bloom_filter.add_list_data(false_negative_urls)
        actual_fprs.append(
            learned_bloom_filter.get_fpr_on_test(threshold, test_df, char_to_idx)
        )
        memory_usages.append(learned_bloom_filter.bloom_filter.get_bloom_filter_size())

    return actual_fprs, memory_usages


def evaluate_partitioned_lbf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    toTrain: bool,
    vocab_size: int,
    char_to_idx: dict,
):
    """
    Evaluate the partitioned learned bloom filter
    for the given target False positive rates
    This function outputs the Actual false positive on the test data
    and total memory use
    """

    partitioned_lbf = PartitionedLearnedBloomFilter(vocab_size, device, NUM_PARTITION)
    print("train_df shape", train_df.shape)
    train_dataset = util.create_dataset(train_df, char_to_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if toTrain:
        partitioned_lbf.fit(train_dataloader, epoch=1)
        partitioned_lbf.save_model(MODEL_FILE_PATH)
    else:
        partitioned_lbf.load_model(MODEL_FILE_PATH)

    actual_fprs = []
    memory_usages = []

    labels, sigmoid_scores = partitioned_lbf.get_sigmoid_scores(train_dataloader)

    key_scores, non_key_scores = partitioned_lbf.get_key_and_non_key_scores(
        labels, sigmoid_scores
    )
    key_scores = [tensor.item() for tensor in key_scores]
    non_key_scores = [tensor.item() for tensor in non_key_scores]

    key_scores_mean = statistics.mean(key_scores)
    key_scores_sd = statistics.stdev(key_scores)

    non_key_scores_mean = statistics.mean(non_key_scores)
    non_key_scores_sd = statistics.stdev(non_key_scores)

    print(key_scores_mean, key_scores_sd)
    print(non_key_scores_mean, non_key_scores_sd)

    # sampled_key_scores = np.random.normal(key_scores_mean, key_scores_sd, 100000)
    # sampled_key_scores = [
    #     item for item in sampled_key_scores if (item >= 0 and item <= 1.0)
    # ]

    # sampled_non_key_scores = np.random.normal(
    #     non_key_scores_mean, non_key_scores_sd, 100000
    # )
    # sampled_non_key_scores = [
    #     item for item in sampled_non_key_scores if (item >= 0 and item <= 1.0)
    # ]
    thresholds = PLBF.get_thresholds(key_scores, non_key_scores, NUM_PARTITION)
    print("Thresholds", thresholds)

    malicious_urls_region_wise = partitioned_lbf.get_fn_urls_region_wise(
        thresholds, train_dataloader
    )
    print("size of partition", end=" ")
    for partition in malicious_urls_region_wise:

        print(len(partition), end=" ")
    print()

    for target_fpr in target_fprs:
        partitioned_lbf.bloom_filters = []
        fprs = PLBF.get_optimal_fpr(key_scores, non_key_scores, thresholds, target_fpr)
        print("False positive rates", fprs)

        for i in range(NUM_PARTITION):
            if fprs[i] == 1:
                break
            partitioned_lbf.bloom_filters.append(
                VanillaBloomFilter(
                    max(10, len(malicious_urls_region_wise[i])), fprs[i]
                )  # giving a buffer of 10 for each Bloom Filter
            )
            partitioned_lbf.bloom_filters[i].add_list_data(
                malicious_urls_region_wise[i]
            )

        num_active_regions = len(partitioned_lbf.bloom_filters)
        print("num_active_regions", num_active_regions)
        threshold = thresholds[num_active_regions - 1]

        actual_fprs.append(
            partitioned_lbf.get_fpr_on_test(threshold, thresholds, test_df, char_to_idx)
        )
        memory_usages.append(partitioned_lbf.get_bloom_filter_size())

    return actual_fprs, memory_usages


if __name__ == "__main__":
    combine_url_df = util.combine_url_dataset()
    train_df, test_df = util.get_train_val_test(combine_url_df)
    char_to_idx, idx_to_char = util.get_char_to_idx_and_reverse(combine_url_df)

    actual_fprs, memory_usages = evaluate_vanilla_bf(train_df, test_df)

    print("************")
    print("Actual FPR and Memory Usage for the Vanilla bloom filter")
    print(actual_fprs, memory_usages)

    vocab_size = util.get_vocab_size(char_to_idx)

    actual_fprs, memory_usages = evaluate_learned_bf(
        train_df, test_df, args.train, vocab_size, char_to_idx
    )
    print("************")
    print("Actual FPR and Memory Usage for the Learned bloom filter")
    print(actual_fprs, memory_usages)

    actual_fprs, memory_usages = evaluate_partitioned_lbf(
        train_df, test_df, args.train, vocab_size, char_to_idx
    )
    print("************")
    print("Actual FPR and Memory Usage for the Partitioned Learned bloom filter")
    print(actual_fprs, memory_usages)
