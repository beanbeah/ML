import librosa
import librosa.display
import numpy as np
from PIL import Image
import numpy as np
import pandas as pd
import autokeras as ak
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class CustomRNNBlock(ak.Block):
    def build(self, hp, inputs=None):
        input_tensor = inputs[0]
        x = tf.keras.layers.Reshape((1, input_tensor.shape[1]))(input_tensor)
        x = ak.RNNBlock(layer_type='lstm').build(hp, inputs=x)
        return x

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

# Load your dataset
num_classes = 7
train = pd.read_csv('train.csv')
audio_files = train['file'].values
audio_files = ['sounds/' + i for i in audio_files]
labels = train['symbol'].values
le = LabelEncoder()
labels = le.fit_transform(labels)

# X: Spectrogram images of the audio files (num_samples, height, width, channels)
# y: Labels for the audio files (num_samples, )
images, features = process_wav(audio_files)
y = labels

# Split the dataset into training and testing sets
X_image_train, X_image_val, X_features_train, X_features_val, y_train, y_val = train_test_split(images, features, y, test_size=0.2, random_state=2023)

# Convert labels to one-hot vectors
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


# Create the AutoModel with custom blocks
inputs_image = ak.Input(shape=X_image_train.shape[1:])
image_norm = ak.Normalization()(inputs_image)
inputs_features = ak.Input(shape=(X_features_train.shape[1],))
#image
x_resnet = ak.ResNetBlock(version="v2")(image_norm)
x_conv = ak.ConvBlock()(image_norm)
#features
x_features_dense = ak.DenseBlock()(inputs_features)  
x_features_rnn = CustomRNNBlock()(x_features_dense)
#merge
x = ak.Merge()([x_resnet, x_conv, x_features_rnn])
x = ak.DenseBlock()(x)
output = ak.ClassificationHead(num_classes=num_classes)(x)

clf = ak.AutoModel(
    inputs=[inputs_image, inputs_features],
    outputs=output,
    overwrite=True,
    max_trials=200
)

# Train the AutoModel
clf.fit(
    [X_image_train, X_features_train], 
    y_train, 
    epochs=50, 
    validation_data=([X_image_val, X_features_val], y_val),
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ],
)

# After training the AutoKeras model
best_model = clf.export_model()

# Save the best model to a file
best_model.save("best_autokeras_model.h5")