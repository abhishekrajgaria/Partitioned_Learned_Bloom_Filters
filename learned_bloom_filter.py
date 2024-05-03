"""
learned_bloom_filter.py: Contains a learned model and a overflow bloom filter
"""

__author__ = "Abhishek Rajgaria, Ashish Tiwari"

import util
import torch
from tqdm import tqdm
import torch.nn as nn
from constants import *
import torch.optim as optim
from torch.utils.data import DataLoader
from character_level_gru import CharacterLevelGRU
from character_level_dataset import CharacterLevelDataset


class LearnedBloomFilter:
    """
    A learned Recurrent Neural network GRU and a overflow bloom filter to keep
    false negative rate of zero
    """

    bloom_filter = None

    def __init__(self, vocab_size: int, device: torch.device) -> None:
        self.model = CharacterLevelGRU(vocab_size, HIDDEN_SIZE, EMBEDDING_DIM)
        self.model.to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.device = device

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
                position=0,
                leave=True,
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

    def get_prediction(self, threshold: float, dataloader: DataLoader):
        predictions = []
        labels = []
        with torch.no_grad():
            for batch in dataloader:
                data_batch, label_tensor, _ = batch
                data_batch = data_batch.to(self.device)
                self.model.to(self.device)
                batch_predictions = self.model(data_batch)
                binary_predictions = (batch_predictions >= threshold).squeeze()

                predictions.extend(binary_predictions.tolist())
                labels.extend(label_tensor.tolist())
        return labels, predictions

    def get_threshold(self, target_fpr: float, dataloader: DataLoader) -> float:
        threshold_floor = THRESHOLD_LOWER_LIMIT
        threshold_ceil = THRESHOLD_UPPER_LIMIT

        temp_fpr = 1
        temp_threshold = 0.5  # default value
        while threshold_floor < threshold_ceil and temp_fpr > (target_fpr / 2):
            temp_threshold = (threshold_ceil + threshold_floor) / 2
            labels, predictions = self.get_prediction(temp_threshold, dataloader)
            # print(labels)
            # print(predictions)
            temp_fpr = util.calculate_fpr(labels, predictions)
            if temp_fpr > target_fpr / 2:
                threshold_floor = temp_threshold
            print("temporary_fpr", temp_fpr)

        return temp_threshold

    def get_fn_urls(self, threshold: float, dataloader: DataLoader) -> list:
        labels = []
        urls = []
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                data_batch, label_tensor, data = batch
                data_batch = data_batch.to(self.device)
                self.model.to(self.device)
                batch_predictions = self.model(data_batch)
                binary_predictions = (batch_predictions >= threshold).squeeze()

                predictions.extend(binary_predictions.tolist())
                labels.extend(label_tensor.tolist())
                urls.extend(data)
                # print("prediction ", predictions)
                # print("labels ", labels)
                # print("urls ", urls)
                # break

        false_negative_url = []
        for i in range(len(predictions)):
            if labels[i] == 1 and predictions[i] == 0:
                false_negative_url.append(urls[i])

        return false_negative_url

    def get_fpr_on_test(self, threshold, test_df, char_to_idx) -> float:

        tests_dataset = CharacterLevelDataset(
            test_df["url"].values, test_df["result"].values, MAX_SEQ_LEN, char_to_idx
        )
        batch_size = 1
        test_dataloader = DataLoader(tests_dataset, batch_size=batch_size, shuffle=True)

        potential_false_postive = []

        negative_cnt = 0
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                data_batch, label, data = batch
                data_batch = data_batch.to(self.device)
                self.model.to(self.device)
                if label != 0:
                    continue
                negative_cnt += 1
                batch_predictions = self.model(data_batch)
                batch_predictions = batch_predictions.tolist()[0][0]
                if batch_predictions >= threshold:
                    potential_false_postive.append(data[0])

        fp_cnt = self.bloom_filter.get_fpr_on_test_list(potential_false_postive)

        return fp_cnt / negative_cnt
