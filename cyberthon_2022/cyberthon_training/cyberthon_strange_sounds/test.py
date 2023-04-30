import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from transformers import ASTForAudioClassification, AutoFeatureExtractor, TrainingArguments, Trainer
from torch.nn.utils.rnn import pad_sequence

class AudioDataset(Dataset):
    def __init__(self, audio_files, labels=None, sr=16000):
        self.audio_files = audio_files
        self.labels = labels
        self.sr = sr
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        audio_file_path = self.audio_files[index]
        waveform, _ = torchaudio.load(audio_file_path)
        waveform = waveform.squeeze(0)
        inputs = self.feature_extractor(waveform, sampling_rate=self.sr, return_tensors="pt")
        input_values = inputs.input_values.squeeze(1)

        if self.labels is not None:
            label = self.labels[index]
            return {"input_values": input_values, "labels": label}
        else:
            return {"input_values": input_values}
   
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    # Load your audio files and labels here
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    #Init Train
    train = pd.read_csv('/home/tech/Desktop/cyberthon/train.csv')
    audio_files = train['file'].values
    audio_files = ['/home/tech/Desktop/cyberthon/sounds/' + i for i in audio_files]
    labels = train['symbol'].values
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    #train test split
    from sklearn.model_selection import train_test_split
    train_files, val_files, train_labels, val_labels = train_test_split(audio_files, labels, test_size=0.2, random_state=2023)
    print("LOADING DATA DONE")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create data loaders
    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)  

    # Train AST model
    model = ASTForAudioClassification.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
    model.to(device)


    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_dir='./logs',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

    # Save model
    trainer.save_model('./results')