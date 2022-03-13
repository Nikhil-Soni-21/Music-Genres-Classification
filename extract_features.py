import librosa.feature as F
import librosa.beat as B
import librosa.effects as E
import numpy as np
import librosa
import csv
import utils as U


# Return feature row that contain all features of an audio file.
def ExtractFeatures(filename, label):
    y, sr = librosa.load(U.DATA_PATH + "\\" + label + "\\" + filename)
    Features = [
        filename,
        *B.tempo(y=y, sr=sr),
        *GetFeatureVector1(F.chroma_stft, y, sr),
        *GetFeatureVector1(F.spectral_centroid, y, sr),
        *GetFeatureVector1(F.spectral_bandwidth, y, sr),
        *GetFeatureVector1(F.spectral_contrast, y, sr),
        *GetFeatureVector1(F.spectral_rolloff, y, sr),
        *GetFeatureVector1(F.tonnetz, y, sr),
        *GetFeatureVector2(E.harmonic, y),
        *GetFeatureVector2(E.percussive, y),
        *GetFeatureVector2(F.spectral_flatness, y),
        *GetFeatureVector2(F.zero_crossing_rate, y),
        *GetFeatureVector2(F.rms, y),
        *GetMfcc(y, sr),
        label
    ]
    return Features


# Get Header Field for CSV
def GetHeaders():
    return [
        'filename',
        'tempo',
        'chroma_stft_mean',
        'chroma_stft_var',
        'spectral_centroid_mean',
        'spectral_centroid_var',
        'spectral_bandwidth_mean',
        'spectral_bandwidth_var',
        'spectral_contrast_mean',
        'spectral_contrast_var',
        'spectral_rolloff_mean',
        'spectral_rolloff_var',
        'tonnetz_mean',
        'tonnetz_var',
        'harmonic_mean',
        'harmonic_var',
        'percussive_mean',
        'percussive_var',
        'spectral_flatness_mean',
        'spectral_flatness_var',
        'zero_crossing_rate_mean',
        'zero_crossing_rate_var',
        'rms_mean',
        'rms_var',
        'mfcc1_mean',
        'mfcc1_var',
        'mfcc2_mean',
        'mfcc2_var',
        'mfcc3_mean',
        'mfcc3_var',
        'mfcc4_mean',
        'mfcc4_var',
        'mfcc5_mean',
        'mfcc5_var',
        'mfcc6_mean',
        'mfcc6_var',
        'mfcc7_mean',
        'mfcc7_var',
        'mfcc8_mean',
        'mfcc8_var',
        'mfcc9_mean',
        'mfcc9_var',
        'mfcc10_mean',
        'mfcc10_var',
        'mfcc11_mean',
        'mfcc11_var',
        'mfcc12_mean',
        'mfcc12_var',
        'mfcc13_mean',
        'mfcc13_var',
        'mfcc14_mean',
        'mfcc14_var',
        'mfcc15_mean',
        'mfcc15_var',
        'mfcc16_mean',
        'mfcc16_var',
        'mfcc17_mean',
        'mfcc17_var',
        'mfcc18_mean',
        'mfcc18_var',
        'mfcc19_mean',
        'mfcc19_var',
        'mfcc20_mean',
        'mfcc20_var',
        'label'
    ]


# Return 20 MFCC for audio file
def GetMfcc(y, sr):
    Features = []
    for mfcc in F.mfcc(y=y, sr=sr):
        Features.append(np.mean(mfcc))
        Features.append(np.var(mfcc))
    return Features


# Method to extract features that requires waveform and samplerate.
def GetFeatureVector1(feature_func, y, sr):
    return [
        np.mean(feature_func(y=y, sr=sr)),
        np.var(feature_func(y=y, sr=sr))
    ]


# Method to extract features that requires only waveform.
def GetFeatureVector2(feature_func, y):
    return [
        np.mean(feature_func(y=y)),
        np.var(feature_func(y=y))
    ]


def SaveFeatures(type, out):
    f = open(out, 'w+', newline='')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(GetHeaders())
    for i in range(len(U.GENRES)):
        Files = U.FILES[i]
        if type == 'train':
            Files = Files[:80]
        else:
            Files = Files[80:]

        for File in Files:
            print('Extracting: ', File)
            writer.writerow(ExtractFeatures(File, U.GENRES[i]))


# SaveFeatures('train', U.CSV_TRAIN)
# SaveFeatures('test', U.CSV_TEST)
