from os import listdir
from random import randint

DATA_PATH_30SEC = 'D:\\Datasets\\GTZAN\\30SEC'

CSV_30SEC = f'{DATA_PATH_30SEC}\\GTZAN.csv'

DATA_PATH_3SEC = 'D:\\Datasets\\GTZAN\\3SEC'

CSV_3SEC = f'{DATA_PATH_3SEC}\\GTZAN.csv'

GENRES = listdir(f'{DATA_PATH_30SEC}\\genres')

FILES_30SEC = [listdir(f'{DATA_PATH_30SEC}\\genres\\{x}') for x in GENRES]

SAMPLE_FILES_30SEC = [x[randint(0, 99)] for x in FILES_30SEC]

FILES_3SEC = [listdir(f'{DATA_PATH_3SEC}\\genres\\{x}') for x in GENRES]

SAMPLE_FILES_3SEC = [x[randint(0, 999)] for x in FILES_3SEC]
