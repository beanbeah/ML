import librosa
import librosa.display
import numpy as np
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import autokeras as ak

class CustomRNNBlock(ak.Block):
    def build(self, hp, inputs=None):
        input_tensor = inputs[0]
        x = tf.keras.layers.Reshape((1, input_tensor.shape[1]))(input_tensor)
        x = ak.RNNBlock(layer_type='lstm').build(hp, inputs=x)
        return x

def upload_CTFSG(token, grader, file):
    import urllib.request, os, json
    urllib.request.urlretrieve('https://raw.githubusercontent.com/alttablabs/ctfsg-utils/master/pyctfsglib.py', './pyctfsglib.py')
    print('Downloaded pyctfsglib.py:', 'pyctfsglib.py' in os.listdir())
    import pyctfsglib as ctfsg
    grader = ctfsg.DSGraderClient(grader, token)
    response = json.loads(grader.submitFile(file))
    os.rename(file, f'{response["multiplier"]}_sklearn_{file[:-4]}.csv')
    return response

def process_wav(files):
    images = []
    features = []
    for file in files:
        if file.endswith(".wav") or file.endswith(".mp3"):
            # Load the audio file and compute the spectrogram
            signal, sr = librosa.load(file, sr=None)

            spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr)
            log_spectrogram = librosa.power_to_db(spectrogram)

            # Convert the spectrogram to an image format (height, width, channels)
            img = librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
            img = img.get_array()
            img = np.uint8((img - img.min()) / (img.max() - img.min()) * 255)
            img = Image.fromarray(img)
            img = img.resize((128, 128))  # Resize the image to a fixed size
            img = np.array(img)[:, :, np.newaxis]  # Add the channel dimension
            images.append(img)

            # Get features
            audio_features = []

            mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
            audio_features.extend(np.mean(mfcc.T, axis=0))
            chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_chroma=12)
            audio_features.extend(np.mean(chroma.T, axis=0))
            spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_bands=6)
            audio_features.extend(np.mean(spectral_contrast.T, axis=0))
            tonnetz = librosa.feature.tonnetz(y=signal, sr=sr)
            audio_features.extend(np.mean(tonnetz.T, axis=0))
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal)
            audio_features.extend(np.mean(zero_crossing_rate.T, axis=0))
            spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr, roll_percent=0.95)
            audio_features.extend(np.mean(spectral_rolloff.T, axis=0))

            features.append(audio_features)

    # Convert the lists to numpy arrays
    images = np.array(images)
    features = np.array(features)
    return images, features

#Clear warnings
import warnings
warnings.filterwarnings('ignore')

# Load the best model
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({
    "CustomRNNBlock": CustomRNNBlock,
})
best_model = tf.keras.models.load_model("best_autokeras_model.h5")

#Init Train
num_classes = 7
train = pd.read_csv('train.csv')
audio_files = train['file'].values
audio_files = ['sounds/' + i for i in audio_files]
labels = train['symbol'].values
le = LabelEncoder()
labels = le.fit_transform(labels)

#Init Test
test = pd.read_csv('test.csv')
test_audio_files = test['file'].values
test_audio_files = ['sounds/' + i for i in test_audio_files]

images_test, features_test = process_wav(test_audio_files)

#Predict
y_pred = best_model.predict([images_test, features_test])
y_pred = np.argmax(y_pred, axis=1)
y_pred = le.inverse_transform(y_pred)

test['symbol'] = y_pred
from datetime import datetime
outFile = f'submission_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
test.to_csv(outFile, index=False)

#Upload
import random
GRADER_URL = random.choice([
  "https://hearmewho01.ds.chals.t.cyberthon23.ctf.sg/",
  "https://hearmewho02.ds.chals.t.cyberthon23.ctf.sg/"
])
token = "NrMxsaIrKbsxNvHNoNbEnljIXTxsWLQYUtnVpHSzyQrqEIPYjXZglMuvDjomTEhd"

print(upload_CTFSG(token, GRADER_URL, outFile))