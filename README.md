# Getting it working on your machine
NOTE: Python 3.9 is required for our project.

1. Clone this repository to your machine and cd to the directory.

        `git clone https://github.com/MegaMind98/4964-project && cd 4964-project/`

2. Create a Python virtual environment. Example:

        `python3 -m venv /path/to/env/`

3. Switch to virtual environment you just created. Example:
   
         `source ./bin/activate`

4. Install all the requirements (they exist in requirements.txt). This might take some time depending on your internet connection and computer specification.

        `pip3 install -r requirements.txt`
    
5. Then, run the following command, please make sure to include --train flag, otherwise the parameters of the model will be random.
    
        `python3 main.py --train`

Alternatively, you can utilize the `run.sh` file to automate the process (this is not actively monitored and might not work). If you chose to use `run.sh`, make sure to update your uID in line 9.

If you run into any problems, please contact authors. Sorry! :'

# Description of each files
main.py - The entry point for the program

util.py - Contains helper functions

vanilla_bloom_filter - Contains bloom filter from bloom-filter2 library

simple_bloom.ipynb - Utilized notebook during check-in report

constants.py - Contains constants used throughout source

character_level_dataset.py - Converts the data into truncated tensor points 
(for urls and lables)

character_level_gru.py - Neural Network Model

learned_bloom_filter.py - Contains a learned model and a overflow bloom 
filter

partitioned_learned_bloom_filter.py - A learned Recurrent Neural network GRU and multiple overflow bloom filter to keep false negative rate of zero. There are total of K bloom filter and are distributed based on the threshold values.

PLBF.py - Provides the threshold value and fpr values to partition the data in different Bloom Filters and their false positive rate (Taken from authors GitHub [repository](https://github.com/kapilvaidya24/PLBF/tree/main?tab=readme-ov-file), Permission has been taken from Professor)

# Folder Structure
```
.
| data
│   | malicious_phish.csv
│   | urldata.csv
| character_level_dataset.py
| partitioned_learned_bloom_filter.py
| character_level_gru.py
| PLBF.py
| constants.py
| README.md
| requirements.txt
| learned_bloom_filter.py
| run.sh
| main.py
| util.py
| model.py
| vanilla_bloom_filter.py
```
