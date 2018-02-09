from keras.models import model_from_json
from preprocess import *
import sounddevice as sd
import librosa

# Second dimension of the feature is dim2
feature_dim_2 = 29
# # Feature dimension
feature_dim_1 = 29
channel = 1
epochs = 400
batch_size = 100
verbose = 1
num_classes = 8

data = None

# Predicts one sample

model = model_from_json(open('model.json').read())
model.load_weights('model_weights.h5')

def predict(np_data):
    sample = process_mfcc(np_data)
    print (sample.shape)
    sample_reshaped = sample.reshape(sample.shape[0], feature_dim_1, feature_dim_2, channel)
    p = model.predict(sample_reshaped)
    d = np.argmax(p)
    return get_labels()[0][d], np.max(p)


def callback(indata, outdata, frames, time, status):
    print (predict(indata.ravel()))


duration = 5.5  # seconds
def live_classify():
    
    with sd.Stream(channels=1, callback=callback, blocksize=16000, samplerate=16000):
        sd.sleep(int(duration * 1000))