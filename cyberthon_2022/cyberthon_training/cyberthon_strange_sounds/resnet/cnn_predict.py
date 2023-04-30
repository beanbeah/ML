'''
Ran on Digital Ocean 
32 vCPUs
128GB / 400GB Disk
($1008/mo)
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models


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

def create_data_loaders(train_files, train_labels, val_files, val_labels):
    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=32)

    return train_loader, val_loader

class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AudioClassifier, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        return x

def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            running_corrects = 0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            val_accuracy = running_corrects.double() / len(val_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    return model

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            predictions.extend(preds.cpu().numpy())

    return predictions

def upload_CTFSG(token, grader, file):
    import urllib.request, os, json
    urllib.request.urlretrieve('https://raw.githubusercontent.com/alttablabs/ctfsg-utils/master/pyctfsglib.py', './pyctfsglib.py')
    print('Downloaded pyctfsglib.py:', 'pyctfsglib.py' in os.listdir())
    import pyctfsglib as ctfsg
    grader = ctfsg.DSGraderClient(grader, token)
    response = json.loads(grader.submitFile(file))
    os.rename(file, f'{response["multiplier"]}_sklearn_{file[:-4]}.csv')
    return response

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test = pd.read_csv('../test.csv')
    test_audio_files = test['file'].values
    test_audio_files = ['../sounds/' + i for i in test_audio_files]

    import pickle
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    test_dataset = AudioDataset(test_audio_files)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=32)

    # Initialize the model
    num_classes = 7  # Change this to the number of classes you have
    model = AudioClassifier(num_classes).to(device)

    # Load the trained model
    model.load_state_dict(torch.load('model.pth'))

    # Run predictions on the test dataset
    predictions = predict(model, test_loader, device)

    # Convert predictions to class labels
    predicted_labels = le.inverse_transform(predictions)

    test['symbol'] = predicted_labels
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