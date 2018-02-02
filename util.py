import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf
import tempfile
import os

def get_labels(path=None):
    path = path or os.getcwd()
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices

def process(filepath):
    f, sr = librosa.load(filepath, sr=48000)
    max16val = np.iinfo(np.int16).max

    return ( 
        librosa.resample( # 16000 hz
            librosa.util.fix_length( # pad
                librosa.effects.trim(  # remove silence
                    librosa.to_mono(f) # mono
                )[0], 
            48000
            ), 
        48000, 
        16000
        ) * max16val
    ).astype(np.int16), 16000


def create_graphs(y, sr):
  
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    plt.subplot(2, 1, 1)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear spectrogram')

    # Or on a logarithmic scale
    plt.subplot(2, 1, 2)
    librosa.display.specshow(D, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log spectrogram')
    return plt


def process_set(path=None):
    path = path or os.getcwd()
    print ("processing files from:", path)
    labels, _ = get_labels(path)

    for label in labels:
        if os.path.isdir(os.path.join(path, label)):
            
            outdir = os.path.join(tempfile.gettempdir(), os.path.basename(path), label)
            outgraphdir = os.path.join(tempfile.gettempdir(), os.path.basename(path)+'_meta', label)
            
            for dir in [outdir, outgraphdir]:
                os.makedirs(dir, exist_ok=True)
            
            wavfiles = [
                os.path.join(path,label,wavfile) for wavfile in 
                os.listdir(os.path.join(path, label))
            ]

            for wavfile in tqdm(wavfiles, "processing - '{}'".format(label)):
                data, sr = process(wavfile)
                
                filename = os.path.basename(wavfile)
                outwav = os.path.join(outdir, filename)
                outpic = os.path.join(outgraphdir, filename.replace('wav', 'png'))

                # save wav
                if os.path.exists(outwav):
                    os.remove(outwav)
                sf.write(outwav, data, sr)

                # save pics
                pic = create_graphs(data, sr)
                pic.savefig(outpic)
                pic.close()

def clean(path):

    """
        used for cleaning training data

        searches wavefile and detects if corresponding
        graph data has been deleted.

        if so, we assume it was bad, so remove wavefile

    """
    path = path or os.getcwd()
    print ("processing files from:", path)
    labels, _ = get_labels(path)
    for label in labels:
        deleted = 0
        if os.path.isdir(os.path.join(path, label)):
            outdir = os.path.join(tempfile.gettempdir(), os.path.basename(path), label)
            outgraphdir = os.path.join(tempfile.gettempdir(), os.path.basename(path)+'_meta', label)

            wavfiles = [
                os.path.join(outdir, wavfile) for wavfile in 
                os.listdir(outdir)
            ]

            for wavfile in tqdm(wavfiles, "processing - '{}'".format(label)):
                filename = os.path.basename(wavfile)
                graphfile = os.path.join(outgraphdir, filename.replace('.wav', '.png'))
                # if graph file has been deleted
                if not os.path.exists(graphfile):
                    os.remove(wavfile)
                    deleted += 1
        print ("deleted {}".format(deleted))
                    

