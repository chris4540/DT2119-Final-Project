from . import features
import scipy
import scipy.signal
import soundfile
import numpy as np

class TIMITFeatureExtractor:
    """
    Simple class to:
    1. transform wave file to MFCC vector
    2. pick up phomes from the transcription file.
    """

    # https://python-speech-features.readthedocs.io/en/latest/
    mfcc_config = {
        # for nfft ~ 512 as 2**9 = 512 just larger than 320 = 20e-3 * 16e3
        "nfft": 512,
        # 20ms = windows length
        "winlen": 20e-3,
        # 10ms = windows sampling rate
        "winstep": 10e-3,
        # number of mfcc values
        "numcep": 13,
        # liftering coefficient used to equalise scale of MFCCs
        "ceplifter": 22,
        # the number of filters in the filterbank, default 26
        "nfilt": 40,
        # MFCCs 1..12 are from DFT, 0 is the log energy component.
        "appendEnergy": True,
    }

    dmfcc_config = {
        # delta MFCC: the number of frames preceding and following for calculating
        # the delta-cepstral
        "ndelta_frames": 3
    }

    def __init__(self, wavfile):
        self.sig, self.sample_freq = self.load_audio(wavfile)


    def get_mfcc_vecs(self):
        win_func = lambda M: self.windows(self, M)
        # calculate mfcc
        mfcc = features.mfcc(
            self.sig, self.sample_freq,
            winfunc=win_func, **self.mfcc_config)

        # calculate delta-mfcc
        dmfcc = features.delta(mfcc, N=self.dmfcc_config['ndelta_frames'])

        # calculate delta-delta-mfcc
        ddmfcc = features.delta(dmfcc, N=self.dmfcc_config['ndelta_frames'])
        ret = np.hstack((mfcc, dmfcc, ddmfcc))
        return ret

    @staticmethod
    def load_audio(filename):
        """
        loads audio data from file using soundfile
        """
        sig, sample_freq = soundfile.read(filename, dtype='int16')
        return sig, sample_freq

    @staticmethod
    def windows(self, frame_size):
        """
        Apply hamming window to the input frames.

        Call back for mfcc lib
        """
        return scipy.signal.hamming(frame_size, sym=False)
