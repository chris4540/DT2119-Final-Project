"""
This is an example to show how to use librosa to do feature extraction.

It is just an example. More work to be done with comparing it to lab1

The feature vector should have:
    12 mfcc features
    12 mfcc 1st order features
    12 mfcc 2nd order features
    1 log energy feature

Windows user:
You need to install ffmpeg

TODO: compare the implementation behind the lab
Ref:
    https://blog.harryfyodor.xyz/2017/10/17/make-mfcc/
    http://mirlab.org/jang/books/audioSignalProcessing/speechFeatureMfcc_chinese.asp?title=12-2%20MFCC
    https://zhuanlan.zhihu.com/p/27416870
    https://librosa.github.io/librosa/generated/librosa.feature.delta.html
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sample_file = librosa.util.example_audio_file()
    y, sr = librosa.load(sample_file, offset=30, duration=5)
    hop_length = 512
    print(y.shape)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=12)
    delta = librosa.feature.delta(mfcc)  # velocity
    delta_delta = librosa.feature.delta(mfcc, order=2) # acceleration
    log_energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
    print("mfcc.shape = ", mfcc.shape)
    print("delta.shape = ", delta.shape)
    print("delta_delta.shape = ", delta_delta.shape)
    # ============================================================================
    plt.subplot(3, 1, 1)
    librosa.display.specshow(mfcc)
    plt.title('MFCC')
    plt.colorbar()
    plt.subplot(3, 1, 2)
    librosa.display.specshow(delta)
    plt.title(r'MFCC-$\Delta$')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    librosa.display.specshow(delta_delta, x_axis='time')
    plt.title(r'MFCC-$\Delta^2$')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("mfcc.png")