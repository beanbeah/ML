import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

def upload_CTFSG(token, grader, file):
    import urllib.request, os, json
    urllib.request.urlretrieve('https://raw.githubusercontent.com/alttablabs/ctfsg-utils/master/pyctfsglib.py', './pyctfsglib.py')
    print('Downloaded pyctfsglib.py:', 'pyctfsglib.py' in os.listdir())
    import pyctfsglib as ctfsg
    grader = ctfsg.DSGraderClient(grader, token)
    response = json.loads(grader.submitFile(file))
    os.rename(file, f'{response["multiplier"]}_sklearn_{file[:-4]}.csv')
    return response

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        se = self.avg_pool(x).view(batch_size, channels)
        se = F.relu(self.fc1(se), inplace=True)
        se = F.sigmoid(self.fc2(se)).view(batch_size, channels, 1, 1)
        return x * se

class AudioClassifier(nn.Module):
    def __init__(self, num_classes, dropout):
        super(AudioClassifier, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes)
        self.dropout = nn.Dropout(dropout)
        self.se = SqueezeExcitation(1536)  # Adjusted for EfficientNet-B3
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.efficientnet.extract_features(x)
        x = self.se(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
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

        # Duplicate single channel to create 3 identical channels
        waveform = waveform.repeat(3, 1, 1)

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

'''
Downloaded pyctfsglib.py: True
DSGraderClient: Successfully Connected!
[SERVER] MOTD: CHECK your USER_TOKEN and GRADER_URL HTTP address! I'm StrangeSounds @ds-hearmewho-alpha-697f5d8fd5-br5fg
ProofOfWork Challenge =>  ('CTFSGRB72517703599a65b08414f4acc2592374', 22)
ProofOfWork Answer Found! =>  3552330
{'challenge': {'name': 'StrangeSounds!'}, 'id': 'clg5to7mm0oau0924l94wyd4b', 'status': 'PARTIALLY_CORRECT', 'multiplier': 0.8886, 'submittedBy': {'username': 'acsi-02'}, 'createdAt': '2023-04-07T00:42:12Z'}
'''

