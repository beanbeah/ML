import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn

def upload_CTFSG(token, grader, file):
    import urllib.request, os, json
    urllib.request.urlretrieve('https://raw.githubusercontent.com/alttablabs/ctfsg-utils/master/pyctfsglib.py', './pyctfsglib.py')
    print('Downloaded pyctfsglib.py:', 'pyctfsglib.py' in os.listdir())
    import pyctfsglib as ctfsg
    grader = ctfsg.DSGraderClient(grader, token)
    response = json.loads(grader.submitFile(file))
    os.rename(file, f'{response["multiplier"]}_sklearn_{file[:-4]}.csv')
    return response

class AudioClassifier(nn.Module):
    def __init__(self, num_classes, num_layers, dropout):
        super(AudioClassifier, self).__init__()
        layers = []
        in_channels = 1
        out_channels = 64

        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels
            out_channels *= 2

        self.conv_layers = nn.Sequential(*layers)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dummy input to calculate fully connected layer's input size
        dummy_input = torch.randn(1, 1, 128, 128)
        dummy_output = self.forward_conv(dummy_input)
        self.fc_input_size = dummy_output.view(-1).size(0)

        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward_conv(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_avg_pool(x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class AudioDataset(Dataset):
    def __init__(self, audio_files, labels=None, target_length=1, sr=16000):
        self.audio_files = audio_files
        self.labels = labels
        self.target_length = target_length
        self.sr = sr
        self.mel_spectrogram = T.MelSpectrogram(sample_rate=self.sr)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file_path = self.audio_files[index]

        waveform, orig_freq = torchaudio.load(audio_file_path)
        waveform = self.preprocess_waveform(waveform, orig_freq)

        if self.labels is not None:
            label = self.labels[index]
            return waveform, label
        else:
            return waveform

    def preprocess_waveform(self, waveform, orig_freq):
        resampler = T.Resample(orig_freq=orig_freq, new_freq=self.sr)
        waveform = resampler(waveform)

        target_samples = self.target_length * self.sr
        waveform_length = waveform.shape[1]

        if waveform_length < target_samples:
            padding_left = (target_samples - waveform_length) // 2
            padding_right = target_samples - waveform_length - padding_left
            waveform = torch.nn.functional.pad(waveform, (padding_left, padding_right))
        elif waveform_length > target_samples:
            crop_left = (waveform_length - target_samples) // 2
            waveform = waveform[:, crop_left:crop_left + target_samples]

        mel_spectrogram = self.mel_spectrogram(waveform)
        return mel_spectrogram

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # Load your audio files and labels here
    import pandas as pd

    #Init Test
    test = pd.read_csv('test.csv')
    test_audio_files = test['file'].values
    test_audio_files = ['sounds/' + i for i in test_audio_files]

    # Load the test dataset
    test_dataset = AudioDataset(test_audio_files)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

    # Other config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the best trained model
    best_trained_model = torch.load('best_model.pth')
    best_trained_model.eval()

    # Load the label encoder
    import pickle
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    # Perform predictions on the test dataset
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = best_trained_model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # Decode the predictions
    y_pred = le.inverse_transform(predictions)
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