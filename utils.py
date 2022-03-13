from os import listdir
from random import randint

# Path of the Audio Files
DATA_PATH = 'D:\\Datasets\\GTZAN\\genres'

# Output File Path for Training Data
CSV_TRAIN = f'D:\\Datasets\\GTZAN\\GTZAN_TRAIN.csv'

# Output File Path for Testing Data
CSV_TEST = f'D:\\Datasets\\GTZAN\\GTZAN_TEST.csv'

# Getting Labels or Genre Types
GENRES = listdir(DATA_PATH)

# Listing All Files of Each Genre as a 2-D List
FILES = [listdir(f'{DATA_PATH}\\{x}') for x in GENRES]

# Contains 10 Random Files from Each Genre for testing Purpose
SAMPLE_FILES = [x[randint(0, 99)] for x in FILES]
