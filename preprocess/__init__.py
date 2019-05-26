from . import features
import scipy
import scipy.signal
import soundfile
import numpy as np
from pathlib import Path

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

        self.wavfile = wavfile
        # load audio file
        self.sig, self.sample_freq = self.load_audio(wavfile)

        # load phone info
        self.phone_info = dict()
        phone_file = Path(wavfile).with_suffix('.PHN')
        self.phone_info['file'] = phone_file
        self.load_phone_info()
        #

    def extract(self):
        self.get_mfcc_vecs()
        self.map_phone_to_features()
        ret = {
            'file': self.wavfile,
            'features': self.features.astype('float32'),
            'phone': self.labels,
        }
        return ret

    def load_phone_info(self):
        starts = list()
        ends = list()
        phones = list()
        with open(self.phone_info['file'], 'r') as f:
            for line in f:
                s, e, ph = line.rstrip().split()
                starts.append(int(s))
                ends.append(int(e))
                phones.append(ph)

        # save them down
        self.phone_info['starts'] = starts
        self.phone_info['ends'] = ends
        self.phone_info['phones'] = phones
        self.phone_info['end_frame'] = ends[-1]

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

        # remove a few features
        nframes = self.get_num_frames()
        assert nframes <= ret.shape[0]
        ret = ret[:nframes, :]
        self.features = ret
        return ret

    def get_windows_length_step(self):
        winlen = self.mfcc_config['winlen']*self.sample_freq
        winstep = self.mfcc_config['winstep']*self.sample_freq
        return winlen, winstep

    def get_num_frames(self):
        """
        Get the number of frames acc. to the phn file
        """
        winlen, winstep = self.get_windows_length_step()
        ret = np.ceil((self.phone_info['end_frame']-winlen) / winstep) + 1
        return int(ret)

    def map_phone_to_features(self):
        # calculate the mid points of mfcc vectors
        winlen, winstep = self.get_windows_length_step()
        mid_pts = np.arange(self.features.shape[0])*winstep + winlen*0.5

        # calculate the tags acc. to mid-points
        x = np.array(mid_pts)
        starts = self.phone_info['starts']
        ends = self.phone_info['ends']
        rets = self.phone_info['phones']
        conds = list()
        for s, e in zip(starts, ends):
            conds.append(np.all([s <= x, x < e], axis=0))
        labels = np.select(conds, rets)
        self.labels = labels
        return labels

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
