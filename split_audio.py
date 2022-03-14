from pydub import AudioSegment
from os import listdir
import utils as U


def SplitAudio():
    ls = []
    for i in range(len(U.GENRES)):
        Files = U.FILES_30SEC[i]
        for F in Files:
            A = AudioSegment.from_wav(
                f'{U.DATA_PATH_30SEC}\\genres\\{U.GENRES[i]}\\{F}')
            print('Splitting File: ', F)
            for j in range(10):
                fn = F[:-3] + str(j)
                seg = A[j*3000: (j+1)*3000]
                seg.export(
                    f'{U.DATA_PATH_3SEC}\\genres\\{U.GENRES[i]}\\{fn}.wav', format='wav')


SplitAudio()
