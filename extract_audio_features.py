import csv
import librosa.feature as F
import librosa
from os import listdir
from numpy import mean
import paths as P

index = 1

F1 = [
    F.chroma_stft,
    F.spectral_centroid,
    F.spectral_bandwidth,
    F.spectral_rolloff,
    F.spectral_contrast,
    F.tempogram,
    F.mfcc,
    F.tonnetz
]

F2 = [
    F.rms,
    F.zero_crossing_rate
]

headers = [
    's.no',
    'file_name',
    'chroma_stft',
    'spectral_centroid',
    'spectral_bandwidth',
    'spectral_rolloff',
    'spectral_contrast',
    'tempogram',
    'mfcc',
    'tonnetz',
    'rms',
    'zero_crossing_rate',
    'genre'
]


def get_feature_vector(y, sr):
    Fe1 = [mean(func(y=y, sr=sr)) for func in F1]
    Fe2 = [mean(func(y=y)) for func in F2]
    return Fe1 + Fe2


def get_features(files, genre, log=False):
    features = []
    global index
    for file in files:
        y, sr = librosa.load(f'{P.DATAPATH}\\{genre}\\{file}')
        features.append([index, file] + get_feature_vector(y, sr) + [genre])
        index += 1
        if log:
            print('Done: ', file)
    return features


def get_files(genre):
    return listdir(f'{P.DATAPATH}\\{genre}')


def get_features_all():
    F = []
    for genre in P.genres:
        F.extend(get_features(get_files(genre), genre, True))
    return F


def save_features():
    f = open(P.CSV_DATAPATH, '+w', newline='')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(headers)
    writer.writerows(get_features_all())
    f.close()


# save_features()
# y, sr = librosa.load(f'{P.DATAPATH}\\blues\\blues.00001.wav')
# print(get_feature_vector(y, sr))
