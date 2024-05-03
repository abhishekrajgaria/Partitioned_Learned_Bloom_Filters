"""
partitioned_learned_bloom_filter.py: Contains a learned model 
and multiple overflow bloom filter
"""

__author__ = "Abhishek Rajgaria, Ashish Tiwari"

import util
import PLBF
import torch
import random
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from constants import *
import torch.optim as optim
from torch.utils.data import DataLoader
from character_level_gru import CharacterLevelGRU
from character_level_dataset import CharacterLevelDataset


class PartitionedLearnedBloomFilter:
    """
    A learned Recurrent Neural network GRU and multiple overflow bloom filter to keep
    false negative rate of zero. There are total of K bloom filter and are distributed
    based on the threshold values.
    """

    bloom_filters = []

    def __init__(self, vocab_size: int, device: torch.device, k: int) -> None:
        self.model = CharacterLevelGRU(vocab_size, HIDDEN_SIZE, EMBEDDING_DIM)
        self.model.to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.device = device
        self.num_partition = k

    def fit(self, dataloader: DataLoader, epoch: int) -> None:
        num_epochs = epoch
        total_batches = len(dataloader)
        running_loss = 0
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            batch_progress = tqdm(
                enumerate(dataloader),
                total=total_batches,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=False,
            )
            for _, (data, labels, _) in batch_progress:
                data = data.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output.squeeze(), labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                epoch_loss += loss.item()

            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / total_batches:.4f}"
            )

        print(f"Final Average Loss: {running_loss / total_batches:.4f}")

    def save_model(self, filepath):
        self.model.to("cpu")
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(
            torch.load(filepath, map_location=torch.device(self.device))
        )
        self.model.eval()

    def get_sigmoid_scores(self, dataloader: DataLoader):
        labels = []
        sigmoid_output = []
        with torch.no_grad():
            for batch in dataloader:
                data_batch, label_tensor, _ = batch
                data_batch = data_batch.to(self.device)
                self.model.to(self.device)
                batch_predictions = self.model(data_batch)
                # print("batch_prediction_shape", batch_predictions.shape)
                # print(batch_predictions.squeeze().tolist())
                sigmoid_output.extend(batch_predictions.squeeze())
                labels.extend(label_tensor.tolist())
        print("# labels", len(labels))
        print("# sigmoid_output", len(sigmoid_output))
        return labels, sigmoid_output

    def get_key_and_non_key_scores(self, labels, sigmoid_scores):
        key_scores = []
        non_key_scores = []

        for label, sigmoid_score in zip(labels, sigmoid_scores):
            if label == 1.0:
                key_scores.append(sigmoid_score)
            else:
                non_key_scores.append(sigmoid_score)

        print("# keys", len(key_scores))
        print("# non keys", len(non_key_scores))
        return key_scores, non_key_scores

    def get_thresholds_and_fprs(self, dataloader: DataLoader, target_fpr: float):
        key_scores = []
        non_key_scores = []
        labels, sigmoid_scores = self.get_sigmoid_scores(dataloader)
        for label, sigmoid_score in zip(labels, sigmoid_scores):
            if label == 1.0:
                key_scores.append(sigmoid_score)
            else:
                non_key_scores.append(sigmoid_score)

        print("# keys", len(key_scores))
        print("# non keys", len(non_key_scores))
        return key_scores, non_key_scores
        # sampled_key_scores = random.sample(key_scores, NUM_SAMPLES)
        # sampled_non_key_scores = random.sample(non_key_scores, NUM_SAMPLES)
        # return PLBF.get_parameter_vals(
        #     key_scores, non_key_scores, target_fpr, self.num_partition
        # )

    def get_region_idx(self, sigmoid_score: float, thresholds: list):
        for i in range(len(thresholds)):
            if sigmoid_score <= thresholds[i]:
                return i

        return 0  # incase of no region found in above loop

    def get_fn_urls_region_wise(self, thresholds: list, dataloader: DataLoader):
        true_positive_urls_region_wise = [[] for _ in range(len(thresholds))]

        labels = []
        urls = []
        sigmoid_output = []
        with torch.no_grad():
            for batch in dataloader:
                data_batch, label_tensor, data = batch
                data_batch = data_batch.to(self.device)
                self.model.to(self.device)
                batch_predictions = self.model(data_batch)

                sigmoid_output.extend(batch_predictions.squeeze())
                labels.extend(label_tensor.tolist())
                urls.extend(data)

        for label, sigmoid_score, url in zip(labels, sigmoid_output, urls):
            if label == 1.0:
                region_idx = self.get_region_idx(sigmoid_score, thresholds)
                true_positive_urls_region_wise[region_idx].append(url)

        return true_positive_urls_region_wise

    def get_fpr_on_test(
        self,
        threshold: float,
        thresholds: list,
        test_df: pd.DataFrame,
        char_to_idx: dict,
    ):
        tests_dataset = CharacterLevelDataset(
            test_df["url"].values, test_df["result"].values, MAX_SEQ_LEN, char_to_idx
        )
        batch_size = 1
        test_dataloader = DataLoader(tests_dataset, batch_size=batch_size, shuffle=True)

        potential_false_postive = []
        tn_cnt = 0
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                data_batch, label, data = batch
                data_batch = data_batch.to(self.device)
                self.model.to(self.device)
                if label != 0:
                    continue
                tn_cnt += 1
                batch_predictions = self.model(data_batch)
                sigmoid_score = batch_predictions.squeeze().item()
                if sigmoid_score >= threshold:
                    potential_false_postive.append(data[0])
                    continue
                region_idx = self.get_region_idx(sigmoid_score, thresholds)

                if data[0] in self.bloom_filters[region_idx].bloom_filter:
                    potential_false_postive.append(data[0])

        fp_cnt = len(potential_false_postive)
        return fp_cnt / tn_cnt

    def get_bloom_filter_size(self):
        size = 0
        for bloom_filter in self.bloom_filters:
            size += bloom_filter.get_bloom_filter_size()

        return size
