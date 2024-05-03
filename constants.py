"""
constants.py: Contains constant value
"""

__author__ = "Abhishek Rajgaria, Ashish Tiwari"

RESULT_COLUMN = "result"
URL_COLUMN = "url"

NUM_PARTITION = 5

MAX_SEQ_LEN = 32
BATCH_SIZE = 64
HIDDEN_SIZE = 16
EMBEDDING_DIM = 32
LEARNING_RATE = 0.001
THRESHOLD_LOWER_LIMIT = 0.05
THRESHOLD_UPPER_LIMIT = 0.75

NUM_SAMPLES = 10000


URL_DATASET_PATH = "./data/urldata.csv"
MALICIOUS_PHISH_DATASET_PATH = "./data/malicious_phish.csv"

MODEL_FILE_PATH = "temp_model.pkl"
