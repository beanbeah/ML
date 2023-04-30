import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

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

def create_data_loaders(train_files, train_labels, val_files, val_labels, batch_size):
    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

    return train_loader, val_loader

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

def train_model(config, checkpoint_dir=None, data_dir=None, num_epochs=30):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_data_loaders(
        train_files, train_labels, val_files, val_labels,
        config["batch_size"],
    )

    model = AudioClassifier(config["num_classes"], config["dropout"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Add learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["lr"], epochs=num_epochs, steps_per_epoch=len(train_loader))

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update learning rate
            lr_scheduler.step()

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

        tune.report(loss=epoch_loss, accuracy=val_accuracy.cpu().numpy())

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            torch.save((model.state_dict(), optimizer.state_dict()), os.path.join(checkpoint_dir, "checkpoint"))

def get_best_trial_custom(trials, metric_name, mode):
    best_trial = None
    best_metric_value = None

    for trial in trials:
        metric_value = trial.last_result[metric_name].item()
        if best_trial is None or (mode == "max" and metric_value > best_metric_value) or (
            mode == "min" and metric_value < best_metric_value
        ):
            best_trial = trial
            best_metric_value = metric_value

    return best_trial

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


    # Set up RayTune experiment
    config = {
        "num_classes": 7,
        "lr": tune.loguniform(1e-6, 1e-4),  # Smaller learning rate range
        "batch_size": tune.choice([8, 16, 32]),  # Smaller batch sizes
        "dropout": tune.uniform(0.1, 0.5),
    }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=90,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"]
    )

    result = tune.run(
        partial(train_model, data_dir = None),
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=config,
        num_samples=350,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="audio_classification_tune"
    )

    best_trial = get_best_trial_custom(result.trials, "accuracy", "max")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    # Save the best model
    best_trained_model = AudioClassifier(best_trial.config["num_classes"], best_trial.config["dropout"]).to(device)
    best_trained_model.to(device)
    best_checkpoint_dir = best_trial.checkpoint.dir_or_data
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    torch.save(best_trained_model, 'best_model.pth')
    
    # Save the label encoder
    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)   

    print("SAVING DONE")

'''
Number of trials: 350/350 (350 TERMINATED)
+-------------------------+------------+-----------------------+--------------+-----------+-------------+-------------+------------+----------------------+
| Trial name              | status     | loc                   |   batch_size |   dropout |          lr |        loss |   accuracy |   training_iteration |
|-------------------------+------------+-----------------------+--------------+-----------+-------------+-------------+------------+----------------------|
| train_model_3dd7d_00000 | TERMINATED | 10.164.33.209:3985895 |            8 |  0.231798 | 1.72531e-06 | 0.0815468   |  0.864286  |                   50 |
| train_model_3dd7d_00001 | TERMINATED | 10.164.33.209:4028611 |           16 |  0.450604 | 1.89034e-06 | 1.95091     |  0.1       |                    1 |
| train_model_3dd7d_00002 | TERMINATED | 10.164.33.209:4029555 |           32 |  0.366755 | 8.85792e-05 | 1.92999     |  0.1       |                    1 |
| train_model_3dd7d_00003 | TERMINATED | 10.164.33.209:4030489 |           32 |  0.370786 | 1.13976e-05 | 1.95796     |  0.05      |                    1 |
| train_model_3dd7d_00004 | TERMINATED | 10.164.33.209:4031426 |            8 |  0.367889 | 3.38403e-05 | 7.90263e-05 |  0.892857  |                   50 |
| train_model_3dd7d_00005 | TERMINATED | 10.164.33.209:4074075 |           32 |  0.128595 | 5.97295e-06 | 1.95968     |  0.114286  |                    2 |
| train_model_3dd7d_00006 | TERMINATED | 10.164.33.209:4075850 |           16 |  0.274372 | 9.53479e-06 | 1.94731     |  0.135714  |                    2 |
| train_model_3dd7d_00007 | TERMINATED | 10.164.33.209:4077641 |           16 |  0.141286 | 8.83104e-05 | 1.93494     |  0.0428571 |                    1 |
| train_model_3dd7d_00008 | TERMINATED | 10.164.33.209:4078584 |           16 |  0.233511 | 8.7247e-06  | 1.94565     |  0.135714  |                    2 |
| train_model_3dd7d_00009 | TERMINATED | 10.164.33.209:4080376 |            8 |  0.443383 | 1.03334e-05 | 1.94899     |  0.107143  |                    1 |
| train_model_3dd7d_00010 | TERMINATED | 10.164.33.209:4081317 |            8 |  0.268295 | 2.04762e-05 | 0.00019116  |  0.857143  |                   32 |
| train_model_3dd7d_00011 | TERMINATED | 10.164.33.209:4108645 |           16 |  0.381636 | 8.83772e-05 | 1.94148     |  0.0785714 |                    1 |
| train_model_3dd7d_00012 | TERMINATED | 10.164.33.209:4109584 |           16 |  0.483765 | 1.38177e-05 | 1.89683     |  0.307143  |                    4 |
| train_model_3dd7d_00013 | TERMINATED | 10.164.33.209:4113076 |           16 |  0.20135  | 2.12633e-06 | 1.93646     |  0.135714  |                    2 |
| train_model_3dd7d_00014 | TERMINATED | 10.164.33.209:4114867 |            8 |  0.347038 | 1.80242e-05 | 0.450935    |  0.778571  |                    8 |
| train_model_3dd7d_00015 | TERMINATED | 10.164.33.209:4121765 |            8 |  0.469613 | 2.24572e-06 | 1.9737      |  0.107143  |                    1 |
| train_model_3dd7d_00016 | TERMINATED | 10.164.33.209:4122706 |           32 |  0.350695 | 2.28993e-06 | 1.95457     |  0.128571  |                    2 |
| train_model_3dd7d_00017 | TERMINATED | 10.164.33.209:4124481 |            8 |  0.459438 | 8.90859e-05 | 0.0615055   |  0.835714  |                   16 |
| train_model_3dd7d_00018 | TERMINATED | 10.164.33.209:4138177 |           32 |  0.313701 | 4.54636e-06 | 1.9051      |  0.192857  |                    4 |
| train_model_3dd7d_00019 | TERMINATED | 10.164.33.209:4141637 |            8 |  0.275918 | 5.63853e-06 | 1.95529     |  0.05      |                    1 |
| train_model_3dd7d_00020 | TERMINATED | 10.164.33.209:4142580 |           16 |  0.237966 | 3.00591e-06 | 1.95589     |  0.135714  |                    2 |
| train_model_3dd7d_00021 | TERMINATED | 10.164.33.209:4144372 |           32 |  0.104465 | 3.10656e-05 | 1.89434     |  0.35      |                    4 |
| train_model_3dd7d_00022 | TERMINATED | 10.164.33.209:4147831 |           32 |  0.237314 | 4.97413e-06 | 1.93651     |  0.1       |                    1 |
| train_model_3dd7d_00023 | TERMINATED | 10.164.33.209:4148770 |            8 |  0.293235 | 6.79681e-06 | 1.90516     |  0.35      |                    4 |
| train_model_3dd7d_00024 | TERMINATED | 10.164.33.209:4152324 |            8 |  0.145465 | 8.21346e-06 | 1.26535     |  0.714286  |                    8 |
| train_model_3dd7d_00025 | TERMINATED | 10.164.33.209:4159220 |           16 |  0.250513 | 5.23356e-05 | 1.94929     |  0.114286  |                    1 |
| train_model_3dd7d_00026 | TERMINATED | 10.164.33.209:4160159 |           32 |  0.329894 | 1.14024e-05 | 1.97359     |  0.1       |                    1 |
| train_model_3dd7d_00027 | TERMINATED | 10.164.33.209:4161094 |           32 |  0.294372 | 2.01793e-06 | 1.94179     |  0.214286  |                    4 |
| train_model_3dd7d_00028 | TERMINATED | 10.164.33.209:4164558 |           32 |  0.438213 | 7.34907e-05 | 1.96388     |  0.114286  |                    1 |
| train_model_3dd7d_00029 | TERMINATED | 10.164.33.209:4165489 |           16 |  0.211754 | 5.75857e-06 | 1.91173     |  0.271429  |                    4 |
| train_model_3dd7d_00030 | TERMINATED | 10.164.33.209:4168995 |           16 |  0.28228  | 1.39708e-06 | 1.96351     |  0.0928571 |                    1 |
| train_model_3dd7d_00031 | TERMINATED | 10.164.33.209:4169937 |           16 |  0.145249 | 8.02743e-06 | 1.90899     |  0.307143  |                    4 |
| train_model_3dd7d_00032 | TERMINATED | 10.164.33.209:4173427 |           32 |  0.464735 | 6.44894e-06 | 1.95897     |  0.214286  |                    4 |
| train_model_3dd7d_00033 | TERMINATED | 10.164.33.209:4176887 |           16 |  0.378752 | 3.2019e-05  | 0.514932    |  0.728571  |                    8 |
| train_model_3dd7d_00034 | TERMINATED | 10.164.33.209:4183780 |            8 |  0.390221 | 4.21275e-06 | 1.94323     |  0.164286  |                    2 |
| train_model_3dd7d_00035 | TERMINATED | 10.164.33.209:4185575 |           16 |  0.215371 | 6.36259e-06 | 1.95428     |  0.185714  |                    2 |
| train_model_3dd7d_00036 | TERMINATED | 10.164.33.209:4187365 |           16 |  0.260271 | 4.7544e-05  | 1.95477     |  0.1       |                    1 |
| train_model_3dd7d_00037 | TERMINATED | 10.164.33.209:4188305 |           16 |  0.386839 | 4.41823e-06 | 1.96543     |  0.0357143 |                    1 |
| train_model_3dd7d_00038 | TERMINATED | 10.164.33.209:4189246 |            8 |  0.476373 | 1.02959e-06 | 1.93779     |  0.128571  |                    2 |
| train_model_3dd7d_00039 | TERMINATED | 10.164.33.209:4191037 |           16 |  0.463399 | 4.50284e-06 | 1.93373     |  0.15      |                    2 |
| train_model_3dd7d_00040 | TERMINATED | 10.164.33.209:4192826 |            8 |  0.37762  | 1.29515e-05 | 1.95066     |  0.0857143 |                    1 |
| train_model_3dd7d_00041 | TERMINATED | 10.164.33.209:4193770 |           32 |  0.100001 | 3.74595e-05 | 1.11875     |  0.671429  |                    8 |
| train_model_3dd7d_00042 | TERMINATED | 10.164.33.209:7072    |           16 |  0.425457 | 5.62562e-05 | 0.0416707   |  0.75      |                   16 |
| train_model_3dd7d_00043 | TERMINATED | 10.164.33.209:20774   |            8 |  0.454214 | 4.74939e-05 | 4.54256e-05 |  0.885714  |                   50 |
| train_model_3dd7d_00044 | TERMINATED | 10.164.33.209:63421   |           16 |  0.368657 | 3.60953e-06 | 1.96103     |  0.15      |                    2 |
| train_model_3dd7d_00045 | TERMINATED | 10.164.33.209:65216   |            8 |  0.362596 | 8.16416e-06 | 1.96126     |  0.05      |                    1 |
| train_model_3dd7d_00046 | TERMINATED | 10.164.33.209:66161   |            8 |  0.220251 | 2.44299e-06 | 1.9338      |  0.164286  |                    2 |
| train_model_3dd7d_00047 | TERMINATED | 10.164.33.209:67979   |            8 |  0.185222 | 2.38285e-06 | 1.92628     |  0.278571  |                    4 |
| train_model_3dd7d_00048 | TERMINATED | 10.164.33.209:71474   |            8 |  0.384729 | 1.2197e-06  | 1.95259     |  0.25      |                    4 |
| train_model_3dd7d_00049 | TERMINATED | 10.164.33.209:74976   |            8 |  0.167629 | 8.8037e-06  | 1.96446     |  0.171429  |                    2 |
| train_model_3dd7d_00050 | TERMINATED | 10.164.33.209:76767   |           32 |  0.355705 | 2.68628e-05 | 1.94869     |  0.1       |                    1 |
| train_model_3dd7d_00051 | TERMINATED | 10.164.33.209:77698   |           32 |  0.461952 | 2.96391e-05 | 1.96807     |  0.121429  |                    1 |
| train_model_3dd7d_00052 | TERMINATED | 10.164.33.209:78630   |           16 |  0.140414 | 4.10042e-05 | 0.291899    |  0.771429  |                    8 |
| train_model_3dd7d_00053 | TERMINATED | 10.164.33.209:85530   |            8 |  0.342043 | 4.53708e-05 | 8.69722e-05 |  0.864286  |                   32 |
| train_model_3dd7d_00054 | TERMINATED | 10.164.33.209:112835  |           16 |  0.483682 | 2.87167e-06 | 1.9464      |  0.05      |                    1 |
| train_model_3dd7d_00055 | TERMINATED | 10.164.33.209:113774  |            8 |  0.217513 | 7.68571e-05 | 1.94181     |  0.0785714 |                    1 |
| train_model_3dd7d_00056 | TERMINATED | 10.164.33.209:114717  |           16 |  0.103205 | 1.72108e-06 | 1.92783     |  0.171429  |                    2 |
| train_model_3dd7d_00057 | TERMINATED | 10.164.33.209:116507  |            8 |  0.4413   | 5.35036e-06 | 1.94397     |  0.0285714 |                    1 |
| train_model_3dd7d_00058 | TERMINATED | 10.164.33.209:117451  |           32 |  0.153649 | 1.39568e-05 | 1.93528     |  0.171429  |                    2 |
| train_model_3dd7d_00059 | TERMINATED | 10.164.33.209:119228  |            8 |  0.353898 | 8.26345e-05 | 1.94456     |  0.0785714 |                    1 |
| train_model_3dd7d_00060 | TERMINATED | 10.164.33.209:120171  |           16 |  0.429067 | 2.29239e-05 | 0.80419     |  0.735714  |                    8 |
| train_model_3dd7d_00061 | TERMINATED | 10.164.33.209:127063  |            8 |  0.10249  | 9.36181e-05 | 1.93098     |  0.0928571 |                    1 |
| train_model_3dd7d_00062 | TERMINATED | 10.164.33.209:128003  |            8 |  0.309516 | 1.73416e-06 | 1.94137     |  0.157143  |                    2 |
| train_model_3dd7d_00063 | TERMINATED | 10.164.33.209:129795  |           32 |  0.104613 | 2.30788e-05 | 1.94382     |  0.0428571 |                    1 |
| train_model_3dd7d_00064 | TERMINATED | 10.164.33.209:130726  |           16 |  0.329948 | 1.17036e-05 | 1.96063     |  0.0428571 |                    1 |
| train_model_3dd7d_00065 | TERMINATED | 10.164.33.209:131668  |           16 |  0.335512 | 3.9407e-06  | 1.95756     |  0.0785714 |                    1 |
| train_model_3dd7d_00066 | TERMINATED | 10.164.33.209:132611  |            8 |  0.40143  | 2.75617e-06 | 1.95298     |  0.107143  |                    1 |
| train_model_3dd7d_00067 | TERMINATED | 10.164.33.209:133554  |           32 |  0.134572 | 2.46807e-06 | 1.92371     |  0.171429  |                    2 |
| train_model_3dd7d_00068 | TERMINATED | 10.164.33.209:135330  |           16 |  0.462887 | 7.93506e-06 | 1.94798     |  0.135714  |                    2 |
| train_model_3dd7d_00069 | TERMINATED | 10.164.33.209:137120  |           16 |  0.20667  | 2.16567e-06 | 1.95888     |  0.0857143 |                    1 |
| train_model_3dd7d_00070 | TERMINATED | 10.164.33.209:138059  |            8 |  0.199017 | 2.38564e-05 | 0.000115078 |  0.878571  |                   50 |
| train_model_3dd7d_00071 | TERMINATED | 10.164.33.209:180698  |            8 |  0.456642 | 8.15946e-05 | 0.186032    |  0.807143  |                   16 |
| train_model_3dd7d_00072 | TERMINATED | 10.164.33.209:194428  |           32 |  0.18042  | 7.625e-05   | 1.93647     |  0.114286  |                    1 |
| train_model_3dd7d_00073 | TERMINATED | 10.164.33.209:195359  |           32 |  0.480196 | 9.88448e-05 | 1.9385      |  0.1       |                    1 |
| train_model_3dd7d_00074 | TERMINATED | 10.164.33.209:196294  |           16 |  0.475437 | 6.12602e-05 | 1.96111     |  0.0642857 |                    1 |
| train_model_3dd7d_00075 | TERMINATED | 10.164.33.209:197234  |           16 |  0.124094 | 7.87522e-05 | 0.225711    |  0.742857  |                    8 |
| train_model_3dd7d_00076 | TERMINATED | 10.164.33.209:204132  |           16 |  0.215324 | 1.19545e-06 | 1.95446     |  0.135714  |                    2 |
| train_model_3dd7d_00077 | TERMINATED | 10.164.33.209:205921  |            8 |  0.318563 | 3.65591e-05 | 0.0435651   |  0.828571  |                   16 |
| train_model_3dd7d_00078 | TERMINATED | 10.164.33.209:219625  |            8 |  0.288731 | 2.4437e-06  | 1.94825     |  0.142857  |                    2 |
| train_model_3dd7d_00079 | TERMINATED | 10.164.33.209:221420  |            8 |  0.497964 | 1.12415e-06 | 1.95064     |  0.164286  |                    2 |
| train_model_3dd7d_00080 | TERMINATED | 10.164.33.209:223225  |            8 |  0.368857 | 4.7275e-06  | 1.94414     |  0.157143  |                    2 |
| train_model_3dd7d_00081 | TERMINATED | 10.164.33.209:225018  |            8 |  0.34704  | 3.18937e-05 | 0.133138    |  0.821429  |                   16 |
| train_model_3dd7d_00082 | TERMINATED | 10.164.33.209:238717  |            8 |  0.151867 | 8.98664e-05 | 0.076575    |  0.828571  |                   16 |
| train_model_3dd7d_00083 | TERMINATED | 10.164.33.209:252428  |           32 |  0.311279 | 1.97246e-05 | 1.95064     |  0.157143  |                    2 |
| train_model_3dd7d_00084 | TERMINATED | 10.164.33.209:254207  |           16 |  0.220418 | 9.80658e-05 | 1.94227     |  0.0928571 |                    1 |
| train_model_3dd7d_00085 | TERMINATED | 10.164.33.209:255148  |            8 |  0.418663 | 2.74658e-06 | 1.95487     |  0.0714286 |                    1 |
| train_model_3dd7d_00086 | TERMINATED | 10.164.33.209:256091  |           32 |  0.136289 | 7.96359e-06 | 1.91235     |  0.264286  |                    4 |
| train_model_3dd7d_00087 | TERMINATED | 10.164.33.209:259555  |           16 |  0.207007 | 2.31305e-05 | 0.742176    |  0.735714  |                    8 |
| train_model_3dd7d_00088 | TERMINATED | 10.164.33.209:266466  |           16 |  0.215097 | 4.14464e-05 | 1.93935     |  0.128571  |                    1 |
| train_model_3dd7d_00089 | TERMINATED | 10.164.33.209:267408  |           16 |  0.4247   | 5.7184e-06  | 1.95684     |  0.15      |                    2 |
| train_model_3dd7d_00090 | TERMINATED | 10.164.33.209:269205  |            8 |  0.130427 | 3.12616e-05 | 0.21532     |  0.792857  |                    8 |
| train_model_3dd7d_00091 | TERMINATED | 10.164.33.209:276106  |            8 |  0.340559 | 1.08359e-06 | 1.93355     |  0.221429  |                    4 |
| train_model_3dd7d_00092 | TERMINATED | 10.164.33.209:279611  |            8 |  0.423848 | 2.44683e-06 | 1.9481      |  0.114286  |                    1 |
| train_model_3dd7d_00093 | TERMINATED | 10.164.33.209:280554  |           16 |  0.179695 | 2.79169e-06 | 1.93944     |  0.15      |                    2 |
| train_model_3dd7d_00094 | TERMINATED | 10.164.33.209:282356  |            8 |  0.27479  | 5.09133e-05 | 1.93357     |  0.107143  |                    1 |
| train_model_3dd7d_00095 | TERMINATED | 10.164.33.209:283299  |           32 |  0.100004 | 3.94487e-06 | 1.94661     |  0.0714286 |                    1 |
| train_model_3dd7d_00096 | TERMINATED | 10.164.33.209:284233  |            8 |  0.256562 | 1.79523e-06 | 1.95241     |  0.157143  |                    2 |
| train_model_3dd7d_00097 | TERMINATED | 10.164.33.209:286028  |            8 |  0.183676 | 1.34781e-05 | 0.000261651 |  0.871429  |                   50 |
| train_model_3dd7d_00098 | TERMINATED | 10.164.33.209:328917  |           32 |  0.154804 | 3.51234e-06 | 1.91672     |  0.235714  |                    4 |
| train_model_3dd7d_00099 | TERMINATED | 10.164.33.209:332382  |           16 |  0.109463 | 4.57022e-06 | 1.95406     |  0.15      |                    2 |
| train_model_3dd7d_00100 | TERMINATED | 10.164.33.209:334175  |            8 |  0.224849 | 7.55183e-06 | 1.95069     |  0.0928571 |                    1 |
| train_model_3dd7d_00101 | TERMINATED | 10.164.33.209:335115  |           16 |  0.460218 | 8.39513e-06 | 1.94829     |  0.178571  |                    2 |
| train_model_3dd7d_00102 | TERMINATED | 10.164.33.209:336908  |            8 |  0.494291 | 2.2328e-05  | 0.000147548 |  0.864286  |                   32 |
| train_model_3dd7d_00103 | TERMINATED | 10.164.33.209:364252  |           16 |  0.22827  | 5.38371e-05 | 1.93485     |  0.121429  |                    1 |
| train_model_3dd7d_00104 | TERMINATED | 10.164.33.209:365192  |           32 |  0.429112 | 1.30311e-05 | 1.91051     |  0.335714  |                    4 |
| train_model_3dd7d_00105 | TERMINATED | 10.164.33.209:368657  |            8 |  0.195637 | 2.41658e-06 | 1.92605     |  0.235714  |                    4 |
| train_model_3dd7d_00106 | TERMINATED | 10.164.33.209:372150  |           16 |  0.371234 | 7.72154e-06 | 1.96614     |  0.121429  |                    1 |
| train_model_3dd7d_00107 | TERMINATED | 10.164.33.209:373092  |            8 |  0.141575 | 1.71331e-06 | 1.9442      |  0.135714  |                    2 |
| train_model_3dd7d_00108 | TERMINATED | 10.164.33.209:374887  |           16 |  0.18628  | 1.87714e-05 | 1.9515      |  0.128571  |                    1 |
| train_model_3dd7d_00109 | TERMINATED | 10.164.33.209:375826  |           16 |  0.101551 | 2.00178e-05 | 1.96224     |  0.128571  |                    1 |
| train_model_3dd7d_00110 | TERMINATED | 10.164.33.209:376770  |           16 |  0.228194 | 8.2989e-06  | 1.94517     |  0.157143  |                    2 |
| train_model_3dd7d_00111 | TERMINATED | 10.164.33.209:378566  |           32 |  0.340758 | 4.49692e-05 | 1.94219     |  0.128571  |                    1 |
| train_model_3dd7d_00112 | TERMINATED | 10.164.33.209:379502  |           32 |  0.422069 | 3.0431e-06  | 1.94607     |  0.114286  |                    1 |
| train_model_3dd7d_00113 | TERMINATED | 10.164.33.209:380434  |            8 |  0.440594 | 3.26037e-06 | 1.96258     |  0.107143  |                    1 |
| train_model_3dd7d_00114 | TERMINATED | 10.164.33.209:381378  |           16 |  0.209893 | 6.30871e-05 | 1.94816     |  0.128571  |                    1 |
| train_model_3dd7d_00115 | TERMINATED | 10.164.33.209:382320  |            8 |  0.204691 | 6.26895e-06 | 1.94075     |  0.185714  |                    2 |
| train_model_3dd7d_00116 | TERMINATED | 10.164.33.209:384116  |           32 |  0.155533 | 7.82581e-05 | 1.95598     |  0.128571  |                    1 |
| train_model_3dd7d_00117 | TERMINATED | 10.164.33.209:385047  |           16 |  0.440436 | 5.08399e-06 | 1.95166     |  0.128571  |                    1 |
| train_model_3dd7d_00118 | TERMINATED | 10.164.33.209:385989  |           32 |  0.208961 | 3.75753e-06 | 1.93956     |  0.178571  |                    2 |
| train_model_3dd7d_00119 | TERMINATED | 10.164.33.209:387769  |           32 |  0.256748 | 2.12094e-06 | 1.94734     |  0.157143  |                    2 |
| train_model_3dd7d_00120 | TERMINATED | 10.164.33.209:389546  |           16 |  0.272373 | 1.54828e-06 | 1.9241      |  0.0785714 |                    1 |
| train_model_3dd7d_00121 | TERMINATED | 10.164.33.209:390486  |           32 |  0.297778 | 3.20984e-06 | 1.92995     |  0.142857  |                    2 |
| train_model_3dd7d_00122 | TERMINATED | 10.164.33.209:392262  |           32 |  0.105842 | 8.05256e-05 | 1.94258     |  0.1       |                    1 |
| train_model_3dd7d_00123 | TERMINATED | 10.164.33.209:393195  |           32 |  0.243096 | 3.85549e-05 | 0.917332    |  0.728571  |                    8 |
| train_model_3dd7d_00124 | TERMINATED | 10.164.33.209:400036  |           16 |  0.477361 | 5.83419e-06 | 1.9495      |  0.142857  |                    2 |
| train_model_3dd7d_00125 | TERMINATED | 10.164.33.209:401832  |           16 |  0.370842 | 1.59784e-05 | 1.94877     |  0.0785714 |                    1 |
| train_model_3dd7d_00126 | TERMINATED | 10.164.33.209:402774  |           32 |  0.289206 | 5.99277e-06 | 1.96962     |  0.121429  |                    1 |
| train_model_3dd7d_00127 | TERMINATED | 10.164.33.209:403710  |            8 |  0.420838 | 1.03577e-06 | 1.92901     |  0.214286  |                    4 |
| train_model_3dd7d_00128 | TERMINATED | 10.164.33.209:407206  |           32 |  0.352904 | 5.22455e-05 | 0.731874    |  0.742857  |                    8 |
| train_model_3dd7d_00129 | TERMINATED | 10.164.33.209:414035  |            8 |  0.214719 | 1.14331e-06 | 1.94696     |  0.0571429 |                    1 |
| train_model_3dd7d_00130 | TERMINATED | 10.164.33.209:414978  |           16 |  0.122171 | 1.20552e-06 | 1.93973     |  0.2       |                    4 |
| train_model_3dd7d_00131 | TERMINATED | 10.164.33.209:418470  |            8 |  0.259164 | 1.90389e-06 | 1.94505     |  0.157143  |                    2 |
| train_model_3dd7d_00132 | TERMINATED | 10.164.33.209:420261  |           32 |  0.103756 | 3.32927e-05 | 1.95145     |  0.0714286 |                    1 |
| train_model_3dd7d_00133 | TERMINATED | 10.164.33.209:421192  |           32 |  0.157622 | 4.20276e-06 | 1.92087     |  0.2       |                    4 |
| train_model_3dd7d_00134 | TERMINATED | 10.164.33.209:424657  |            8 |  0.224306 | 1.80508e-05 | 1.95119     |  0.178571  |                    2 |
| train_model_3dd7d_00135 | TERMINATED | 10.164.33.209:426451  |           16 |  0.492752 | 8.11768e-06 | 1.95703     |  0.142857  |                    2 |
| train_model_3dd7d_00136 | TERMINATED | 10.164.33.209:428244  |            8 |  0.352679 | 5.80093e-05 | 4.66106e-05 |  0.871429  |                   50 |
| train_model_3dd7d_00137 | TERMINATED | 10.164.33.209:470920  |            8 |  0.481653 | 2.11014e-06 | 1.92486     |  0.25      |                    4 |
| train_model_3dd7d_00138 | TERMINATED | 10.164.33.209:474426  |            8 |  0.145605 | 7.90086e-06 | 1.88856     |  0.378571  |                    4 |
| train_model_3dd7d_00139 | TERMINATED | 10.164.33.209:477935  |            8 |  0.192249 | 4.00224e-06 | 1.89217     |  0.264286  |                    4 |
| train_model_3dd7d_00140 | TERMINATED | 10.164.33.209:481430  |            8 |  0.401159 | 3.03729e-06 | 1.94519     |  0.128571  |                    1 |
| train_model_3dd7d_00141 | TERMINATED | 10.164.33.209:482373  |            8 |  0.319437 | 3.76746e-06 | 1.94032     |  0.05      |                    1 |
| train_model_3dd7d_00142 | TERMINATED | 10.164.33.209:483317  |           16 |  0.365048 | 2.71389e-05 | 1.97098     |  0.114286  |                    1 |
| train_model_3dd7d_00143 | TERMINATED | 10.164.33.209:484260  |            8 |  0.428992 | 1.31106e-05 | 0.00024896  |  0.907143  |                   50 |
| train_model_3dd7d_00144 | TERMINATED | 10.164.33.209:526906  |            8 |  0.30791  | 2.22441e-06 | 1.9586      |  0.0714286 |                    1 |
| train_model_3dd7d_00145 | TERMINATED | 10.164.33.209:527848  |           32 |  0.411329 | 5.56103e-05 | 1.95907     |  0.0857143 |                    1 |
| train_model_3dd7d_00146 | TERMINATED | 10.164.33.209:528781  |           32 |  0.164219 | 2.52877e-06 | 1.9656      |  0.142857  |                    2 |
| train_model_3dd7d_00147 | TERMINATED | 10.164.33.209:530557  |           16 |  0.127103 | 1.39015e-06 | 1.94948     |  0.135714  |                    2 |
| train_model_3dd7d_00148 | TERMINATED | 10.164.33.209:532350  |           32 |  0.177504 | 1.50736e-05 | 1.95451     |  0.171429  |                    2 |
| train_model_3dd7d_00149 | TERMINATED | 10.164.33.209:534126  |           16 |  0.356179 | 8.92427e-06 | 1.9319      |  0.15      |                    2 |
| train_model_3dd7d_00150 | TERMINATED | 10.164.33.209:535924  |            8 |  0.407433 | 5.44583e-05 | 4.99348e-05 |  0.885714  |                   50 |
| train_model_3dd7d_00151 | TERMINATED | 10.164.33.209:578571  |            8 |  0.195104 | 8.31149e-05 | 0.0450929   |  0.835714  |                   16 |
| train_model_3dd7d_00152 | TERMINATED | 10.164.33.209:592285  |           32 |  0.372887 | 8.48109e-05 | 1.94623     |  0.121429  |                    1 |
| train_model_3dd7d_00153 | TERMINATED | 10.164.33.209:593216  |           32 |  0.496336 | 6.53329e-05 | 0.466009    |  0.764286  |                    8 |
| train_model_3dd7d_00154 | TERMINATED | 10.164.33.209:600049  |           16 |  0.421131 | 1.51292e-05 | 1.97132     |  0.0714286 |                    1 |
| train_model_3dd7d_00155 | TERMINATED | 10.164.33.209:600991  |           16 |  0.277068 | 1.72137e-05 | 1.87111     |  0.421429  |                    4 |
| train_model_3dd7d_00156 | TERMINATED | 10.164.33.209:604493  |           32 |  0.481527 | 4.22605e-05 | 0.897894    |  0.721429  |                    8 |
| train_model_3dd7d_00157 | TERMINATED | 10.164.33.209:611327  |           32 |  0.413557 | 8.8527e-06  | 1.93561     |  0.0642857 |                    1 |
| train_model_3dd7d_00158 | TERMINATED | 10.164.33.209:612262  |            8 |  0.388739 | 1.01363e-05 | 1.91106     |  0.364286  |                    4 |
| train_model_3dd7d_00159 | TERMINATED | 10.164.33.209:615758  |           32 |  0.215081 | 1.94315e-05 | 1.93788     |  0.171429  |                    2 |
| train_model_3dd7d_00160 | TERMINATED | 10.164.33.209:617534  |            8 |  0.205136 | 1.40653e-06 | 1.93417     |  0.235714  |                    4 |
| train_model_3dd7d_00161 | TERMINATED | 10.164.33.209:621042  |            8 |  0.460606 | 1.02185e-05 | 1.94833     |  0.135714  |                    2 |
| train_model_3dd7d_00162 | TERMINATED | 10.164.33.209:622835  |            8 |  0.133167 | 1.44268e-06 | 1.94207     |  0.157143  |                    2 |
| train_model_3dd7d_00163 | TERMINATED | 10.164.33.209:624630  |           16 |  0.107484 | 4.78816e-06 | 1.94099     |  0.171429  |                    2 |
| train_model_3dd7d_00164 | TERMINATED | 10.164.33.209:626422  |           16 |  0.400347 | 2.70963e-05 | 1.94416     |  0.107143  |                    1 |
| train_model_3dd7d_00165 | TERMINATED | 10.164.33.209:627364  |           32 |  0.375104 | 1.45124e-06 | 1.91639     |  0.242857  |                    4 |
| train_model_3dd7d_00166 | TERMINATED | 10.164.33.209:630824  |           16 |  0.303178 | 5.65659e-05 | 1.92758     |  0.121429  |                    1 |
| train_model_3dd7d_00167 | TERMINATED | 10.164.33.209:631766  |           32 |  0.341001 | 4.09019e-06 | 1.94805     |  0.107143  |                    1 |
| train_model_3dd7d_00168 | TERMINATED | 10.164.33.209:632701  |           16 |  0.26487  | 2.23882e-05 | 1.96595     |  0.0714286 |                    1 |
| train_model_3dd7d_00169 | TERMINATED | 10.164.33.209:633644  |           16 |  0.21166  | 3.61501e-06 | 1.92512     |  0.235714  |                    4 |
| train_model_3dd7d_00170 | TERMINATED | 10.164.33.209:637145  |           32 |  0.18914  | 9.94239e-06 | 1.91828     |  0.221429  |                    4 |
| train_model_3dd7d_00171 | TERMINATED | 10.164.33.209:640608  |           16 |  0.348396 | 2.64899e-05 | 1.96428     |  0.121429  |                    1 |
| train_model_3dd7d_00172 | TERMINATED | 10.164.33.209:641550  |           32 |  0.131507 | 5.24602e-05 | 1.94823     |  0.128571  |                    1 |
| train_model_3dd7d_00173 | TERMINATED | 10.164.33.209:642488  |           32 |  0.277893 | 4.92512e-05 | 1.94893     |  0.1       |                    1 |
| train_model_3dd7d_00174 | TERMINATED | 10.164.33.209:643420  |            8 |  0.432828 | 5.67219e-05 | 4.96665e-05 |  0.885714  |                   50 |
| train_model_3dd7d_00175 | TERMINATED | 10.164.33.209:686091  |           32 |  0.346683 | 4.7293e-05  | 1.97255     |  0.128571  |                    1 |
| train_model_3dd7d_00176 | TERMINATED | 10.164.33.209:687023  |            8 |  0.177193 | 8.35989e-06 | 1.24166     |  0.742857  |                    8 |
| train_model_3dd7d_00177 | TERMINATED | 10.164.33.209:693920  |           32 |  0.336602 | 9.05235e-06 | 1.93776     |  0.15      |                    2 |
| train_model_3dd7d_00178 | TERMINATED | 10.164.33.209:695696  |           32 |  0.466388 | 3.30323e-05 | 1.93723     |  0.121429  |                    1 |
| train_model_3dd7d_00179 | TERMINATED | 10.164.33.209:696627  |           16 |  0.410006 | 2.71406e-05 | 1.94373     |  0.0642857 |                    1 |
| train_model_3dd7d_00180 | TERMINATED | 10.164.33.209:697568  |           16 |  0.410515 | 9.3084e-06  | 1.96377     |  0.0571429 |                    1 |
| train_model_3dd7d_00181 | TERMINATED | 10.164.33.209:698509  |           16 |  0.279892 | 3.73933e-05 | 1.95269     |  0.114286  |                    1 |
| train_model_3dd7d_00182 | TERMINATED | 10.164.33.209:699451  |           32 |  0.421295 | 4.52515e-06 | 1.95616     |  0.1       |                    1 |
| train_model_3dd7d_00183 | TERMINATED | 10.164.33.209:700384  |           16 |  0.194677 | 3.98938e-06 | 1.96012     |  0.121429  |                    1 |
| train_model_3dd7d_00184 | TERMINATED | 10.164.33.209:701324  |            8 |  0.282475 | 7.72428e-06 | 1.89416     |  0.321429  |                    4 |
| train_model_3dd7d_00185 | TERMINATED | 10.164.33.209:704815  |            8 |  0.294957 | 5.16714e-05 | 8.01379e-05 |  0.864286  |                   32 |
| train_model_3dd7d_00186 | TERMINATED | 10.164.33.209:732138  |           32 |  0.347547 | 7.28645e-05 | 1.96132     |  0.0785714 |                    1 |
| train_model_3dd7d_00187 | TERMINATED | 10.164.33.209:733070  |            8 |  0.474398 | 5.63066e-06 | 1.95153     |  0.142857  |                    2 |
| train_model_3dd7d_00188 | TERMINATED | 10.164.33.209:734877  |           16 |  0.359476 | 7.77193e-05 | 0.209807    |  0.785714  |                    8 |
| train_model_3dd7d_00189 | TERMINATED | 10.164.33.209:741777  |           16 |  0.240478 | 2.00237e-06 | 1.94654     |  0.0928571 |                    1 |
| train_model_3dd7d_00190 | TERMINATED | 10.164.33.209:742722  |           32 |  0.38592  | 2.81966e-06 | 1.93346     |  0.207143  |                    4 |
| train_model_3dd7d_00191 | TERMINATED | 10.164.33.209:746181  |            8 |  0.218583 | 1.81997e-05 | 0.000157731 |  0.885714  |                   50 |
| train_model_3dd7d_00192 | TERMINATED | 10.164.33.209:788834  |            8 |  0.314393 | 1.44825e-06 | 1.95122     |  0.0928571 |                    1 |
| train_model_3dd7d_00193 | TERMINATED | 10.164.33.209:789776  |           32 |  0.313619 | 5.77281e-05 | 0.617961    |  0.728571  |                    8 |
| train_model_3dd7d_00194 | TERMINATED | 10.164.33.209:796606  |           32 |  0.319367 | 6.58496e-05 | 1.93733     |  0.121429  |                    1 |
| train_model_3dd7d_00195 | TERMINATED | 10.164.33.209:797541  |            8 |  0.399688 | 3.43262e-05 | 8.56114e-05 |  0.878571  |                   50 |
| train_model_3dd7d_00196 | TERMINATED | 10.164.33.209:840188  |           16 |  0.473151 | 9.55109e-06 | 1.96981     |  0.107143  |                    1 |
| train_model_3dd7d_00197 | TERMINATED | 10.164.33.209:841129  |           32 |  0.221693 | 9.54544e-05 | 1.95282     |  0.171429  |                    2 |
| train_model_3dd7d_00198 | TERMINATED | 10.164.33.209:842908  |           16 |  0.302761 | 4.15433e-06 | 1.9667      |  0.0928571 |                    1 |
| train_model_3dd7d_00199 | TERMINATED | 10.164.33.209:843849  |            8 |  0.408215 | 4.74732e-06 | 1.9534      |  0.0428571 |                    1 |
| train_model_3dd7d_00200 | TERMINATED | 10.164.33.209:844794  |           32 |  0.376334 | 5.90251e-06 | 1.96135     |  0.128571  |                    1 |
| train_model_3dd7d_00201 | TERMINATED | 10.164.33.209:845729  |            8 |  0.174734 | 1.73754e-05 | 0.000223314 |  0.871429  |                   32 |
| train_model_3dd7d_00202 | TERMINATED | 10.164.33.209:873092  |            8 |  0.225574 | 9.88201e-06 | 0.977267    |  0.785714  |                    8 |
| train_model_3dd7d_00203 | TERMINATED | 10.164.33.209:880009  |           32 |  0.427956 | 2.39465e-06 | 1.95396     |  0.171429  |                    2 |
| train_model_3dd7d_00204 | TERMINATED | 10.164.33.209:881788  |           32 |  0.440413 | 9.74425e-05 | 1.96135     |  0.121429  |                    1 |
| train_model_3dd7d_00205 | TERMINATED | 10.164.33.209:882720  |           32 |  0.145319 | 3.63299e-06 | 1.95541     |  0.157143  |                    2 |
| train_model_3dd7d_00206 | TERMINATED | 10.164.33.209:884501  |           16 |  0.114349 | 7.79117e-05 | 1.94405     |  0.114286  |                    1 |
| train_model_3dd7d_00207 | TERMINATED | 10.164.33.209:885444  |            8 |  0.19651  | 7.46926e-05 | 3.82864e-05 |  0.878571  |                   50 |
| train_model_3dd7d_00208 | TERMINATED | 10.164.33.209:928127  |           32 |  0.243725 | 2.60141e-06 | 1.94766     |  0.0428571 |                    1 |
| train_model_3dd7d_00209 | TERMINATED | 10.164.33.209:929058  |           16 |  0.107015 | 1.87429e-05 | 1.95711     |  0.0857143 |                    1 |
| train_model_3dd7d_00210 | TERMINATED | 10.164.33.209:929998  |            8 |  0.302831 | 1.77994e-05 | 1.95132     |  0.0642857 |                    1 |
| train_model_3dd7d_00211 | TERMINATED | 10.164.33.209:930940  |            8 |  0.303375 | 4.7513e-05  | 7.06385e-05 |  0.871429  |                   32 |
| train_model_3dd7d_00212 | TERMINATED | 10.164.33.209:958298  |           16 |  0.229881 | 1.95074e-06 | 1.96163     |  0.114286  |                    1 |
| train_model_3dd7d_00213 | TERMINATED | 10.164.33.209:959239  |           16 |  0.467694 | 2.71116e-06 | 1.96384     |  0.135714  |                    2 |
| train_model_3dd7d_00214 | TERMINATED | 10.164.33.209:961035  |            8 |  0.276771 | 1.05724e-06 | 1.97634     |  0.121429  |                    1 |
| train_model_3dd7d_00215 | TERMINATED | 10.164.33.209:961978  |            8 |  0.448638 | 2.09479e-05 | 0.000140792 |  0.892857  |                   50 |
| train_model_3dd7d_00216 | TERMINATED | 10.164.33.209:1004636 |           32 |  0.420424 | 1.5996e-06  | 1.94524     |  0.15      |                    2 |
| train_model_3dd7d_00217 | TERMINATED | 10.164.33.209:1006413 |            8 |  0.29907  | 1.06483e-06 | 1.95588     |  0.114286  |                    1 |
| train_model_3dd7d_00218 | TERMINATED | 10.164.33.209:1007356 |            8 |  0.35659  | 3.33218e-06 | 1.95244     |  0.107143  |                    1 |
| train_model_3dd7d_00219 | TERMINATED | 10.164.33.209:1008300 |            8 |  0.368872 | 3.38449e-05 | 9.09336e-05 |  0.892857  |                   50 |
| train_model_3dd7d_00220 | TERMINATED | 10.164.33.209:1050984 |           16 |  0.434196 | 1.65219e-05 | 1.94517     |  0.0785714 |                    1 |
| train_model_3dd7d_00221 | TERMINATED | 10.164.33.209:1051930 |            8 |  0.262719 | 1.06333e-05 | 1.87156     |  0.407143  |                    4 |
| train_model_3dd7d_00222 | TERMINATED | 10.164.33.209:1055431 |            8 |  0.466381 | 1.84214e-05 | 0.407492    |  0.814286  |                    8 |
| train_model_3dd7d_00223 | TERMINATED | 10.164.33.209:1062329 |            8 |  0.125632 | 3.13758e-06 | 1.93977     |  0.114286  |                    1 |
| train_model_3dd7d_00224 | TERMINATED | 10.164.33.209:1063273 |           16 |  0.115919 | 3.45891e-05 | 1.948       |  0.171429  |                    2 |
| train_model_3dd7d_00225 | TERMINATED | 10.164.33.209:1065066 |           32 |  0.383743 | 1.02724e-06 | 1.95719     |  0.2       |                    4 |
| train_model_3dd7d_00226 | TERMINATED | 10.164.33.209:1068528 |           16 |  0.135237 | 6.64898e-05 | 1.94528     |  0.128571  |                    1 |
| train_model_3dd7d_00227 | TERMINATED | 10.164.33.209:1069472 |           16 |  0.149464 | 6.55382e-06 | 1.95223     |  0.157143  |                    2 |
| train_model_3dd7d_00228 | TERMINATED | 10.164.33.209:1071274 |           16 |  0.297688 | 2.08937e-05 | 1.94008     |  0.178571  |                    2 |
| train_model_3dd7d_00229 | TERMINATED | 10.164.33.209:1073066 |           32 |  0.201473 | 1.31987e-06 | 1.9483      |  0.0714286 |                    1 |
| train_model_3dd7d_00230 | TERMINATED | 10.164.33.209:1074002 |           16 |  0.302782 | 7.47691e-05 | 0.213231    |  0.778571  |                    8 |
| train_model_3dd7d_00231 | TERMINATED | 10.164.33.209:1080896 |           32 |  0.239752 | 5.56861e-06 | 1.96264     |  0.15      |                    2 |
| train_model_3dd7d_00232 | TERMINATED | 10.164.33.209:1082671 |            8 |  0.329459 | 6.62505e-06 | 1.96529     |  0.114286  |                    1 |
| train_model_3dd7d_00233 | TERMINATED | 10.164.33.209:1083614 |           32 |  0.329136 | 2.40971e-06 | 1.94914     |  0.135714  |                    2 |
| train_model_3dd7d_00234 | TERMINATED | 10.164.33.209:1085394 |           32 |  0.222391 | 3.33533e-05 | 1.96884     |  0.142857  |                    2 |
| train_model_3dd7d_00235 | TERMINATED | 10.164.33.209:1087177 |           32 |  0.310971 | 3.14724e-05 | 1.92045     |  0.142857  |                    2 |
| train_model_3dd7d_00236 | TERMINATED | 10.164.33.209:1088961 |           16 |  0.206423 | 6.44964e-06 | 1.95274     |  0.121429  |                    1 |
| train_model_3dd7d_00237 | TERMINATED | 10.164.33.209:1089901 |           16 |  0.357724 | 2.64384e-06 | 1.96013     |  0.114286  |                    1 |
| train_model_3dd7d_00238 | TERMINATED | 10.164.33.209:1090841 |           16 |  0.272121 | 1.71405e-05 | 1.87271     |  0.421429  |                    4 |
| train_model_3dd7d_00239 | TERMINATED | 10.164.33.209:1094332 |           16 |  0.159338 | 5.74124e-05 | 1.93379     |  0.107143  |                    1 |
| train_model_3dd7d_00240 | TERMINATED | 10.164.33.209:1095275 |           16 |  0.1985   | 2.46672e-05 | 1.95859     |  0.0571429 |                    1 |
| train_model_3dd7d_00241 | TERMINATED | 10.164.33.209:1096215 |            8 |  0.26699  | 1.62257e-06 | 1.95916     |  0.114286  |                    1 |
| train_model_3dd7d_00242 | TERMINATED | 10.164.33.209:1097156 |            8 |  0.466639 | 7.86955e-06 | 1.93996     |  0.185714  |                    2 |
| train_model_3dd7d_00243 | TERMINATED | 10.164.33.209:1098950 |           32 |  0.388893 | 3.01137e-06 | 1.9579      |  0.121429  |                    1 |
| train_model_3dd7d_00244 | TERMINATED | 10.164.33.209:1099887 |            8 |  0.265529 | 3.97868e-05 | 6.62715e-05 |  0.9       |                   50 |
| train_model_3dd7d_00245 | TERMINATED | 10.164.33.209:1142586 |            8 |  0.459588 | 1.36477e-06 | 1.93206     |  0.135714  |                    2 |
| train_model_3dd7d_00246 | TERMINATED | 10.164.33.209:1144379 |           16 |  0.433592 | 2.1664e-05  | 1.96341     |  0.121429  |                    1 |
| train_model_3dd7d_00247 | TERMINATED | 10.164.33.209:1145320 |           16 |  0.185449 | 2.40013e-06 | 1.94171     |  0.121429  |                    1 |
| train_model_3dd7d_00248 | TERMINATED | 10.164.33.209:1146261 |           32 |  0.179753 | 4.25934e-06 | 1.9373      |  0.128571  |                    1 |
| train_model_3dd7d_00249 | TERMINATED | 10.164.33.209:1147194 |           32 |  0.280874 | 9.94412e-05 | 0.256031    |  0.771429  |                    8 |
| train_model_3dd7d_00250 | TERMINATED | 10.164.33.209:1154025 |            8 |  0.270362 | 2.94188e-05 | 0.036699    |  0.842857  |                   16 |
| train_model_3dd7d_00251 | TERMINATED | 10.164.33.209:1167731 |           32 |  0.377572 | 8.83923e-05 | 0.272532    |  0.757143  |                    8 |
| train_model_3dd7d_00252 | TERMINATED | 10.164.33.209:1174573 |            8 |  0.250184 | 2.48269e-06 | 1.94938     |  0.135714  |                    2 |
| train_model_3dd7d_00253 | TERMINATED | 10.164.33.209:1176371 |            8 |  0.166551 | 7.68502e-06 | 1.94926     |  0.0785714 |                    1 |
| train_model_3dd7d_00254 | TERMINATED | 10.164.33.209:1177313 |           16 |  0.105468 | 4.94833e-05 | 0.24443     |  0.778571  |                    8 |
| train_model_3dd7d_00255 | TERMINATED | 10.164.33.209:1184206 |           32 |  0.239293 | 6.25305e-05 | 1.84431     |  0.457143  |                    4 |
| train_model_3dd7d_00256 | TERMINATED | 10.164.33.209:1187673 |           32 |  0.120286 | 3.19397e-05 | 1.88796     |  0.464286  |                    4 |
| train_model_3dd7d_00257 | TERMINATED | 10.164.33.209:1191135 |           32 |  0.157659 | 8.24749e-05 | 1.9414      |  0.128571  |                    1 |
| train_model_3dd7d_00258 | TERMINATED | 10.164.33.209:1192068 |            8 |  0.485156 | 2.92523e-06 | 1.94517     |  0.164286  |                    2 |
| train_model_3dd7d_00259 | TERMINATED | 10.164.33.209:1193865 |           16 |  0.121124 | 1.92008e-06 | 1.92947     |  0.185714  |                    2 |
| train_model_3dd7d_00260 | TERMINATED | 10.164.33.209:1195657 |           16 |  0.41461  | 3.24028e-05 | 0.505634    |  0.742857  |                    8 |
| train_model_3dd7d_00261 | TERMINATED | 10.164.33.209:1202550 |           32 |  0.111058 | 7.3668e-05  | 1.9381      |  0.121429  |                    1 |
| train_model_3dd7d_00262 | TERMINATED | 10.164.33.209:1203484 |           32 |  0.294101 | 2.85199e-06 | 1.95891     |  0.0928571 |                    1 |
| train_model_3dd7d_00263 | TERMINATED | 10.164.33.209:1204419 |            8 |  0.169632 | 2.34055e-05 | 1.9379      |  0.0785714 |                    1 |
| train_model_3dd7d_00264 | TERMINATED | 10.164.33.209:1205369 |           16 |  0.390475 | 1.67742e-06 | 1.93979     |  0.135714  |                    2 |
| train_model_3dd7d_00265 | TERMINATED | 10.164.33.209:1207168 |           16 |  0.11968  | 5.6424e-06  | 1.95962     |  0.107143  |                    1 |
| train_model_3dd7d_00266 | TERMINATED | 10.164.33.209:1208112 |           32 |  0.126614 | 4.13943e-05 | 1.95067     |  0.107143  |                    1 |
| train_model_3dd7d_00267 | TERMINATED | 10.164.33.209:1209045 |           16 |  0.429695 | 7.86041e-05 | 0.172066    |  0.778571  |                    8 |
| train_model_3dd7d_00268 | TERMINATED | 10.164.33.209:1215937 |           16 |  0.289079 | 2.45159e-06 | 1.96922     |  0.1       |                    1 |
| train_model_3dd7d_00269 | TERMINATED | 10.164.33.209:1216878 |           16 |  0.263209 | 3.66704e-06 | 1.93417     |  0.221429  |                    4 |
| train_model_3dd7d_00270 | TERMINATED | 10.164.33.209:1220372 |           16 |  0.145634 | 4.73056e-05 | 0.309875    |  0.785714  |                    8 |
| train_model_3dd7d_00271 | TERMINATED | 10.164.33.209:1227266 |            8 |  0.45532  | 5.80702e-06 | 1.91539     |  0.321429  |                    4 |
| train_model_3dd7d_00272 | TERMINATED | 10.164.33.209:1230762 |           16 |  0.192018 | 5.53135e-06 | 1.95623     |  0.0142857 |                    1 |
| train_model_3dd7d_00273 | TERMINATED | 10.164.33.209:1231702 |           32 |  0.225282 | 3.71283e-06 | 1.94806     |  0.221429  |                    4 |
| train_model_3dd7d_00274 | TERMINATED | 10.164.33.209:1235164 |           32 |  0.415524 | 3.32693e-06 | 1.96203     |  0.0928571 |                    1 |
| train_model_3dd7d_00275 | TERMINATED | 10.164.33.209:1236097 |           16 |  0.311625 | 6.09036e-06 | 1.94565     |  0.114286  |                    1 |
| train_model_3dd7d_00276 | TERMINATED | 10.164.33.209:1237038 |            8 |  0.332389 | 1.64047e-05 | 0.00018913  |  0.907143  |                   50 |
| train_model_3dd7d_00277 | TERMINATED | 10.164.33.209:1279684 |            8 |  0.491759 | 7.98289e-06 | 1.89062     |  0.45      |                    4 |
| train_model_3dd7d_00278 | TERMINATED | 10.164.33.209:1283177 |            8 |  0.221757 | 1.33151e-05 | 1.95484     |  0.171429  |                    2 |
| train_model_3dd7d_00279 | TERMINATED | 10.164.33.209:1284976 |            8 |  0.255168 | 1.81288e-05 | 0.000136237 |  0.892857  |                   50 |
| train_model_3dd7d_00280 | TERMINATED | 10.164.33.209:1327610 |           16 |  0.13257  | 2.97332e-06 | 1.96503     |  0.1       |                    1 |
| train_model_3dd7d_00281 | TERMINATED | 10.164.33.209:1328551 |           16 |  0.441171 | 8.23164e-05 | 0.209888    |  0.764286  |                    8 |
| train_model_3dd7d_00282 | TERMINATED | 10.164.33.209:1335460 |           16 |  0.402984 | 2.05236e-05 | 1.85879     |  0.464286  |                    4 |
| train_model_3dd7d_00283 | TERMINATED | 10.164.33.209:1338952 |           16 |  0.418667 | 2.97559e-05 | 1.96284     |  0.157143  |                    2 |
| train_model_3dd7d_00284 | TERMINATED | 10.164.33.209:1340742 |           16 |  0.201765 | 4.96189e-05 | 1.95516     |  0.0714286 |                    1 |
| train_model_3dd7d_00285 | TERMINATED | 10.164.33.209:1341683 |            8 |  0.437012 | 1.41949e-06 | 1.95392     |  0.221429  |                    4 |
| train_model_3dd7d_00286 | TERMINATED | 10.164.33.209:1345181 |           32 |  0.196327 | 2.26752e-05 | 1.96333     |  0.107143  |                    1 |
| train_model_3dd7d_00287 | TERMINATED | 10.164.33.209:1346113 |           16 |  0.23242  | 2.48418e-05 | 1.95393     |  0.0428571 |                    1 |
| train_model_3dd7d_00288 | TERMINATED | 10.164.33.209:1347053 |           16 |  0.345842 | 4.97084e-05 | 1.95703     |  0.107143  |                    1 |
| train_model_3dd7d_00289 | TERMINATED | 10.164.33.209:1347995 |           16 |  0.338293 | 1.95466e-06 | 1.95673     |  0.0642857 |                    1 |
| train_model_3dd7d_00290 | TERMINATED | 10.164.33.209:1348936 |           32 |  0.375584 | 4.44548e-05 | 1.9467      |  0.107143  |                    1 |
| train_model_3dd7d_00291 | TERMINATED | 10.164.33.209:1349877 |            8 |  0.334072 | 9.48903e-06 | 1.96506     |  0.1       |                    1 |
| train_model_3dd7d_00292 | TERMINATED | 10.164.33.209:1350820 |           32 |  0.208181 | 6.9796e-06  | 1.96309     |  0.0857143 |                    1 |
| train_model_3dd7d_00293 | TERMINATED | 10.164.33.209:1351752 |            8 |  0.394696 | 5.79628e-06 | 1.92744     |  0.185714  |                    2 |
| train_model_3dd7d_00294 | TERMINATED | 10.164.33.209:1353550 |           32 |  0.395845 | 9.09902e-06 | 1.94852     |  0.0928571 |                    1 |
| train_model_3dd7d_00295 | TERMINATED | 10.164.33.209:1354484 |            8 |  0.352804 | 1.14504e-05 | 1.87279     |  0.428571  |                    4 |
| train_model_3dd7d_00296 | TERMINATED | 10.164.33.209:1357977 |           32 |  0.360439 | 5.91103e-06 | 1.95591     |  0.1       |                    1 |
| train_model_3dd7d_00297 | TERMINATED | 10.164.33.209:1358908 |           32 |  0.114495 | 8.31398e-06 | 1.95096     |  0.157143  |                    2 |
| train_model_3dd7d_00298 | TERMINATED | 10.164.33.209:1360682 |           16 |  0.378963 | 2.33687e-05 | 0.735175    |  0.685714  |                    8 |
| train_model_3dd7d_00299 | TERMINATED | 10.164.33.209:1367581 |            8 |  0.102234 | 2.34113e-05 | 0.000138874 |  0.871429  |                   32 |
| train_model_3dd7d_00300 | TERMINATED | 10.164.33.209:1394903 |           32 |  0.138465 | 1.3164e-06  | 1.9622      |  0.0428571 |                    1 |
| train_model_3dd7d_00301 | TERMINATED | 10.164.33.209:1395834 |            8 |  0.233051 | 2.77774e-06 | 1.96329     |  0.128571  |                    1 |
| train_model_3dd7d_00302 | TERMINATED | 10.164.33.209:1396776 |           16 |  0.107781 | 7.82912e-06 | 1.94953     |  0.114286  |                    1 |
| train_model_3dd7d_00303 | TERMINATED | 10.164.33.209:1397717 |           16 |  0.138245 | 6.16187e-06 | 1.92402     |  0.157143  |                    2 |
| train_model_3dd7d_00304 | TERMINATED | 10.164.33.209:1399510 |            8 |  0.222047 | 2.35073e-05 | 0.000121133 |  0.885714  |                   50 |
| train_model_3dd7d_00305 | TERMINATED | 10.164.33.209:1442150 |           16 |  0.106161 | 2.05587e-06 | 1.94424     |  0.142857  |                    2 |
| train_model_3dd7d_00306 | TERMINATED | 10.164.33.209:1443940 |           16 |  0.48687  | 7.66223e-05 | 1.95034     |  0.0714286 |                    1 |
| train_model_3dd7d_00307 | TERMINATED | 10.164.33.209:1444882 |           16 |  0.474211 | 6.07818e-05 | 1.94104     |  0.1       |                    1 |
| train_model_3dd7d_00308 | TERMINATED | 10.164.33.209:1445825 |            8 |  0.464519 | 1.07488e-06 | 1.95192     |  0.114286  |                    1 |
| train_model_3dd7d_00309 | TERMINATED | 10.164.33.209:1446770 |            8 |  0.483722 | 3.70229e-06 | 1.90765     |  0.307143  |                    4 |
| train_model_3dd7d_00310 | TERMINATED | 10.164.33.209:1450267 |            8 |  0.497018 | 6.00796e-06 | 1.88354     |  0.407143  |                    4 |
| train_model_3dd7d_00311 | TERMINATED | 10.164.33.209:1453762 |           32 |  0.374557 | 9.64803e-06 | 1.93394     |  0.128571  |                    1 |
| train_model_3dd7d_00312 | TERMINATED | 10.164.33.209:1454695 |           32 |  0.266302 | 7.50032e-05 | 1.95202     |  0.0857143 |                    1 |
| train_model_3dd7d_00313 | TERMINATED | 10.164.33.209:1455633 |           32 |  0.246    | 2.06521e-05 | 1.9675      |  0.142857  |                    2 |
| train_model_3dd7d_00314 | TERMINATED | 10.164.33.209:1457408 |           32 |  0.368116 | 7.28887e-05 | 1.95001     |  0.0785714 |                    1 |
| train_model_3dd7d_00315 | TERMINATED | 10.164.33.209:1458342 |           32 |  0.3729   | 6.43385e-06 | 1.96034     |  0.121429  |                    1 |
| train_model_3dd7d_00316 | TERMINATED | 10.164.33.209:1459277 |           32 |  0.310378 | 2.97165e-05 | 1.92839     |  0.178571  |                    2 |
| train_model_3dd7d_00317 | TERMINATED | 10.164.33.209:1461054 |            8 |  0.17636  | 1.62679e-05 | 1.94727     |  0.114286  |                    1 |
| train_model_3dd7d_00318 | TERMINATED | 10.164.33.209:1461997 |           16 |  0.299153 | 1.68299e-05 | 1.9568      |  0.121429  |                    1 |
| train_model_3dd7d_00319 | TERMINATED | 10.164.33.209:1462940 |           16 |  0.350202 | 4.64168e-06 | 1.97069     |  0.114286  |                    1 |
| train_model_3dd7d_00320 | TERMINATED | 10.164.33.209:1463882 |           32 |  0.11017  | 1.53915e-06 | 1.95692     |  0.128571  |                    2 |
| train_model_3dd7d_00321 | TERMINATED | 10.164.33.209:1465667 |           32 |  0.304452 | 1.51106e-06 | 1.97121     |  0.0928571 |                    1 |
| train_model_3dd7d_00322 | TERMINATED | 10.164.33.209:1466600 |            8 |  0.402373 | 1.56026e-05 | 1.87822     |  0.435714  |                    4 |
| train_model_3dd7d_00323 | TERMINATED | 10.164.33.209:1470094 |           16 |  0.49886  | 5.28263e-06 | 1.95383     |  0.157143  |                    2 |
| train_model_3dd7d_00324 | TERMINATED | 10.164.33.209:1471884 |           16 |  0.487437 | 4.24282e-06 | 1.95028     |  0.171429  |                    2 |
| train_model_3dd7d_00325 | TERMINATED | 10.164.33.209:1473675 |            8 |  0.324439 | 2.64394e-06 | 1.95768     |  0.05      |                    1 |
| train_model_3dd7d_00326 | TERMINATED | 10.164.33.209:1474617 |           16 |  0.408431 | 1.59511e-05 | 1.96349     |  0.0785714 |                    1 |
| train_model_3dd7d_00327 | TERMINATED | 10.164.33.209:1475557 |            8 |  0.342232 | 8.89018e-06 | 1.89327     |  0.357143  |                    4 |
| train_model_3dd7d_00328 | TERMINATED | 10.164.33.209:1479052 |            8 |  0.357117 | 1.79546e-05 | 1.79056     |  0.457143  |                    4 |
| train_model_3dd7d_00329 | TERMINATED | 10.164.33.209:1482544 |            8 |  0.11858  | 5.27638e-06 | 1.89667     |  0.292857  |                    4 |
| train_model_3dd7d_00330 | TERMINATED | 10.164.33.209:1486035 |            8 |  0.273755 | 4.18366e-05 | 1.95016     |  0.114286  |                    1 |
| train_model_3dd7d_00331 | TERMINATED | 10.164.33.209:1486975 |           16 |  0.355604 | 3.90284e-05 | 1.9597      |  0.128571  |                    1 |
| train_model_3dd7d_00332 | TERMINATED | 10.164.33.209:1487917 |           32 |  0.424836 | 4.96581e-06 | 1.95334     |  0.1       |                    1 |
| train_model_3dd7d_00333 | TERMINATED | 10.164.33.209:1488851 |           16 |  0.322943 | 1.55215e-06 | 1.94485     |  0.107143  |                    1 |
| train_model_3dd7d_00334 | TERMINATED | 10.164.33.209:1489793 |           16 |  0.316387 | 4.80136e-05 | 0.27559     |  0.714286  |                    8 |
| train_model_3dd7d_00335 | TERMINATED | 10.164.33.209:1496685 |           32 |  0.325944 | 6.00733e-06 | 1.94825     |  0.15      |                    2 |
| train_model_3dd7d_00336 | TERMINATED | 10.164.33.209:1498460 |            8 |  0.225929 | 4.38078e-06 | 1.95378     |  0.171429  |                    2 |
| train_model_3dd7d_00337 | TERMINATED | 10.164.33.209:1500251 |           16 |  0.274747 | 1.30323e-06 | 1.95568     |  0.0642857 |                    1 |
| train_model_3dd7d_00338 | TERMINATED | 10.164.33.209:1501191 |            8 |  0.487598 | 1.37687e-06 | 1.94131     |  0.192857  |                    2 |
| train_model_3dd7d_00339 | TERMINATED | 10.164.33.209:1502991 |           16 |  0.166022 | 2.78495e-06 | 1.9391      |  0.135714  |                    2 |
| train_model_3dd7d_00340 | TERMINATED | 10.164.33.209:1504782 |           16 |  0.195505 | 5.44486e-06 | 1.93676     |  0.157143  |                    2 |
| train_model_3dd7d_00341 | TERMINATED | 10.164.33.209:1506573 |           32 |  0.492638 | 3.01461e-05 | 1.94423     |  0.192857  |                    2 |
| train_model_3dd7d_00342 | TERMINATED | 10.164.33.209:1508347 |           16 |  0.297411 | 1.90081e-06 | 1.97219     |  0.0428571 |                    1 |
| train_model_3dd7d_00343 | TERMINATED | 10.164.33.209:1509286 |           32 |  0.488282 | 1.9891e-06  | 1.92422     |  0.242857  |                    4 |
| train_model_3dd7d_00344 | TERMINATED | 10.164.33.209:1512747 |           16 |  0.320804 | 2.16052e-05 | 1.9526      |  0.142857  |                    2 |
| train_model_3dd7d_00345 | TERMINATED | 10.164.33.209:1514538 |           32 |  0.227184 | 4.84802e-05 | 1.95768     |  0.121429  |                    1 |
| train_model_3dd7d_00346 | TERMINATED | 10.164.33.209:1515469 |           16 |  0.346673 | 1.8253e-05  | 1.9522      |  0.121429  |                    1 |
| train_model_3dd7d_00347 | TERMINATED | 10.164.33.209:1516411 |            8 |  0.377544 | 2.47917e-05 | 1.94446     |  0.121429  |                    1 |
| train_model_3dd7d_00348 | TERMINATED | 10.164.33.209:1517353 |           16 |  0.362526 | 8.01807e-06 | 1.94627     |  0.107143  |                    1 |
| train_model_3dd7d_00349 | TERMINATED | 10.164.33.209:1518294 |           16 |  0.299843 | 1.69971e-06 | 1.95697     |  0.0928571 |                    1 |
+-------------------------+------------+-----------------------+--------------+-----------+-------------+-------------+------------+----------------------+


Best trial config: {'num_classes': 7, 'lr': 1.3110565615215219e-05, 'batch_size': 8, 'dropout': 0.42899241856544}
Best trial final validation loss: 0.0002489599126420217
Best trial final validation accuracy: 0.9071428571428571
Loaded pretrained weights for efficientnet-b3
'''