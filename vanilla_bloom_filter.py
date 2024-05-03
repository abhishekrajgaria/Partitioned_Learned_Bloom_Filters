"""
vanilla_bloom_filter.py: contains bloom filter from bloom-filter2
library.
"""

__author__ = "Abhishek Rajgaria, Ashish Tiwari"

import pandas as pd
from bloom_filter2 import BloomFilter
from constants import RESULT_COLUMN, URL_COLUMN


class VanillaBloomFilter:
    """
    vanilla bloom filter is the normal
    bloom filter which only uses has function
    """

    def __init__(self, max_element: int, error_rate: float) -> BloomFilter:
        self.bloom_filter = BloomFilter(max_element, error_rate)

    def add_data(self, df: pd.DataFrame):
        """
        Add maclicious url from the df to the bloom filter
        """
        for _, row in df[df[RESULT_COLUMN] == 1].iterrows():
            self.bloom_filter.add(row[URL_COLUMN])
    
    def add_list_data(self, list: list):
        for item in list:
            self.bloom_filter.add(item)

    def get_fpr_on_test(self, test_df: pd.DataFrame) -> float:
        """
        Computes the False positive rate on the df for benign URLs
        """
        fp_cnt = 0
        tn_cnt = 0
        for _, row in test_df[test_df[RESULT_COLUMN] == 0].iterrows():
            if row[URL_COLUMN] in self.bloom_filter:
                fp_cnt += 1
            tn_cnt+=1
        return fp_cnt / tn_cnt

    def get_fpr_on_test_list(self, list: list) -> float:
        """
        Computes the False positive rate on the df for benign URLs
        """
        fp = 0
        for item in list:
            if(item in self.bloom_filter):
                fp+=1

        return fp

    def get_bloom_filter_size(self):
        """
        provides size of the bloom filter in Mb
        """
        return self.bloom_filter.num_bits_m / (1024 * 1024)
