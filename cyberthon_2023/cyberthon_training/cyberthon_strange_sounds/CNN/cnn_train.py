import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from functools import partial

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

def create_data_loaders(train_files, train_labels, val_files, val_labels, batch_size):
    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)

    return train_loader, val_loader

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
        dummy_output = self._forward_conv(dummy_input)
        self.fc_input_size = dummy_output.view(-1).size(0)

        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def _forward_conv(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_avg_pool(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_model(config, checkpoint_dir=None, data_dir=None, num_epochs=30):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_data_loaders(
        train_files, train_labels, val_files, val_labels,
        config["batch_size"],
    )

    model = AudioClassifier(config["num_classes"], config["num_layers"], config["dropout"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

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
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([16, 32, 64]),
        "num_layers": tune.choice([3, 4, 5, 6]),
        "dropout": tune.uniform(0.1, 0.5),
    }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=30,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"]
    )

    result = tune.run(
        partial(train_model, data_dir = None),
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        num_samples=300,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="audio_classification_tune"
    )

    best_trial = get_best_trial_custom(result.trials, "accuracy", "max")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    # Save the best model
    best_trained_model = AudioClassifier(best_trial.config["num_classes"], best_trial.config["num_layers"], best_trial.config["dropout"]).to(device)
    best_trained_model.to(device)
    best_checkpoint_dir = best_trial.checkpoint.dir_or_data
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    torch.save(best_trained_model, 'best_model.pth')

    # Save the label encoder
    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    # Save the config used for training
    import json
    with open('config.json', 'w') as f:
        json.dump(best_trial.config, f)


    print("SAVING DONE")

    '''
    Number of trials: 300/300 (300 TERMINATED)
+-------------------------+------------+-----------------------+--------------+-----------+-------------+--------------+-------------+------------+----------------------+
| Trial name              | status     | loc                   |   batch_size |   dropout |          lr |   num_layers |        loss |   accuracy |   training_iteration |
|-------------------------+------------+-----------------------+--------------+-----------+-------------+--------------+-------------+------------+----------------------|
| train_model_a34bb_00000 | TERMINATED | 10.164.33.209:1820856 |           16 |  0.494967 | 0.000609742 |            4 | 0.048263    |  0.8       |                   30 |
| train_model_a34bb_00001 | TERMINATED | 10.164.33.209:1846475 |           32 |  0.113594 | 3.17973e-05 |            6 | 0.20665     |  0.778571  |                    8 |
| train_model_a34bb_00002 | TERMINATED | 10.164.33.209:1853306 |           64 |  0.345681 | 0.000832307 |            3 | 1.77791     |  0.414286  |                    1 |
| train_model_a34bb_00003 | TERMINATED | 10.164.33.209:1854233 |           32 |  0.227022 | 0.000292943 |            5 | 0.210615    |  0.771429  |                    8 |
| train_model_a34bb_00004 | TERMINATED | 10.164.33.209:1861064 |           16 |  0.3937   | 0.000397347 |            6 | 1.31118     |  0.528571  |                    2 |
| train_model_a34bb_00005 | TERMINATED | 10.164.33.209:1862869 |           16 |  0.450009 | 8.59894e-05 |            6 | 1.52833     |  0.0857143 |                    1 |
| train_model_a34bb_00006 | TERMINATED | 10.164.33.209:1863809 |           32 |  0.196477 | 0.000261684 |            3 | 1.51311     |  0.507143  |                    2 |
| train_model_a34bb_00007 | TERMINATED | 10.164.33.209:1865584 |           16 |  0.336222 | 6.55092e-05 |            4 | 1.74654     |  0.428571  |                    1 |
| train_model_a34bb_00008 | TERMINATED | 10.164.33.209:1866523 |           64 |  0.426907 | 2.02251e-05 |            6 | 1.8155      |  0.335714  |                    1 |
| train_model_a34bb_00009 | TERMINATED | 10.164.33.209:1867449 |           32 |  0.367285 | 1.73508e-05 |            3 | 1.91776     |  0.15      |                    1 |
| train_model_a34bb_00010 | TERMINATED | 10.164.33.209:1868380 |           32 |  0.21175  | 0.000532603 |            3 | 0.630914    |  0.65      |                    4 |
| train_model_a34bb_00011 | TERMINATED | 10.164.33.209:1871842 |           64 |  0.475637 | 0.000674907 |            3 | 1.80395     |  0.25      |                    1 |
| train_model_a34bb_00012 | TERMINATED | 10.164.33.209:1872767 |           16 |  0.152142 | 0.000394404 |            3 | 1.06886     |  0.578571  |                    2 |
| train_model_a34bb_00013 | TERMINATED | 10.164.33.209:1874556 |           16 |  0.293089 | 1.45567e-05 |            5 | 1.77335     |  0.2       |                    1 |
| train_model_a34bb_00014 | TERMINATED | 10.164.33.209:1875497 |           64 |  0.119833 | 6.46376e-05 |            5 | 1.34965     |  0.528571  |                    2 |
| train_model_a34bb_00015 | TERMINATED | 10.164.33.209:1877262 |           64 |  0.491861 | 0.000963083 |            3 | 1.80076     |  0.228571  |                    1 |
| train_model_a34bb_00016 | TERMINATED | 10.164.33.209:1878188 |           32 |  0.197013 | 2.16149e-05 |            4 | 1.83843     |  0.357143  |                    1 |
| train_model_a34bb_00017 | TERMINATED | 10.164.33.209:1879118 |           16 |  0.179186 | 4.99908e-05 |            4 | 0.402895    |  0.742857  |                    8 |
| train_model_a34bb_00018 | TERMINATED | 10.164.33.209:1886010 |           16 |  0.17474  | 0.000508406 |            4 | 0.316901    |  0.692857  |                    8 |
| train_model_a34bb_00019 | TERMINATED | 10.164.33.209:1892896 |           64 |  0.397771 | 0.000384586 |            6 | 1.75408     |  0.414286  |                    1 |
| train_model_a34bb_00020 | TERMINATED | 10.164.33.209:1893823 |           64 |  0.102662 | 3.92833e-05 |            4 | 1.84411     |  0.228571  |                    1 |
| train_model_a34bb_00021 | TERMINATED | 10.164.33.209:1894756 |           64 |  0.360612 | 7.38544e-05 |            6 | 1.63969     |  0.3       |                    1 |
| train_model_a34bb_00022 | TERMINATED | 10.164.33.209:1895683 |           16 |  0.419433 | 5.82402e-05 |            3 | 1.79226     |  0.421429  |                    2 |
| train_model_a34bb_00023 | TERMINATED | 10.164.33.209:1897477 |           64 |  0.454608 | 0.000490481 |            3 | 1.82572     |  0.271429  |                    1 |
| train_model_a34bb_00024 | TERMINATED | 10.164.33.209:1898406 |           64 |  0.104476 | 8.88642e-05 |            6 | 0.226594    |  0.778571  |                   16 |
| train_model_a34bb_00025 | TERMINATED | 10.164.33.209:1911909 |           16 |  0.122973 | 3.18464e-05 |            5 | 0.427215    |  0.728571  |                    8 |
| train_model_a34bb_00026 | TERMINATED | 10.164.33.209:1918794 |           64 |  0.454333 | 2.20848e-05 |            6 | 1.86538     |  0.135714  |                    1 |
| train_model_a34bb_00027 | TERMINATED | 10.164.33.209:1919723 |           64 |  0.411935 | 0.000344404 |            4 | 0.877109    |  0.635714  |                    4 |
| train_model_a34bb_00028 | TERMINATED | 10.164.33.209:1923172 |           64 |  0.416542 | 0.000366535 |            3 | 1.83579     |  0.257143  |                    1 |
| train_model_a34bb_00029 | TERMINATED | 10.164.33.209:1924100 |           64 |  0.427558 | 1.18684e-05 |            4 | 1.96242     |  0.178571  |                    1 |
| train_model_a34bb_00030 | TERMINATED | 10.164.33.209:1925031 |           32 |  0.106754 | 7.34898e-05 |            4 | 0.90949     |  0.685714  |                    4 |
| train_model_a34bb_00031 | TERMINATED | 10.164.33.209:1928493 |           64 |  0.26049  | 0.000126594 |            5 | 1.5987      |  0.271429  |                    1 |
| train_model_a34bb_00032 | TERMINATED | 10.164.33.209:1929421 |           64 |  0.283787 | 7.82929e-05 |            3 | 1.90793     |  0.2       |                    1 |
| train_model_a34bb_00033 | TERMINATED | 10.164.33.209:1930352 |           64 |  0.165709 | 0.000397464 |            6 | 1.54764     |  0.378571  |                    1 |
| train_model_a34bb_00034 | TERMINATED | 10.164.33.209:1931278 |           16 |  0.127354 | 1.06457e-05 |            3 | 1.91594     |  0.185714  |                    1 |
| train_model_a34bb_00035 | TERMINATED | 10.164.33.209:1932218 |           16 |  0.222457 | 0.000386247 |            5 | 0.227184    |  0.742857  |                    8 |
| train_model_a34bb_00036 | TERMINATED | 10.164.33.209:1939106 |           16 |  0.276798 | 0.000144096 |            6 | 0.0152189   |  0.821429  |                   30 |
| train_model_a34bb_00037 | TERMINATED | 10.164.33.209:1964726 |           16 |  0.124099 | 0.000239893 |            6 | 0.0910562   |  0.785714  |                   16 |
| train_model_a34bb_00038 | TERMINATED | 10.164.33.209:1978444 |           64 |  0.155139 | 1.75945e-05 |            3 | 1.92429     |  0.2       |                    1 |
| train_model_a34bb_00039 | TERMINATED | 10.164.33.209:1979372 |           64 |  0.380803 | 2.13124e-05 |            4 | 1.88073     |  0.185714  |                    1 |
| train_model_a34bb_00040 | TERMINATED | 10.164.33.209:1980303 |           16 |  0.409808 | 0.000832644 |            3 | 1.02761     |  0.535714  |                    2 |
| train_model_a34bb_00041 | TERMINATED | 10.164.33.209:1982093 |           32 |  0.443941 | 3.34086e-05 |            4 | 1.85978     |  0.314286  |                    1 |
| train_model_a34bb_00042 | TERMINATED | 10.164.33.209:1983029 |           16 |  0.244611 | 8.46618e-05 |            5 | 0.258402    |  0.771429  |                    8 |
| train_model_a34bb_00043 | TERMINATED | 10.164.33.209:1989917 |           64 |  0.193587 | 1.29852e-05 |            6 | 1.63302     |  0.478571  |                    2 |
| train_model_a34bb_00044 | TERMINATED | 10.164.33.209:1991685 |           32 |  0.390745 | 2.21734e-05 |            5 | 1.80898     |  0.185714  |                    1 |
| train_model_a34bb_00045 | TERMINATED | 10.164.33.209:1992618 |           16 |  0.350369 | 1.81566e-05 |            5 | 1.70369     |  0.242857  |                    1 |
| train_model_a34bb_00046 | TERMINATED | 10.164.33.209:1993556 |           32 |  0.415434 | 0.000738764 |            5 | 1.0814      |  0.578571  |                    2 |
| train_model_a34bb_00047 | TERMINATED | 10.164.33.209:1995327 |           16 |  0.382523 | 0.000959999 |            6 | 1.30813     |  0.542857  |                    2 |
| train_model_a34bb_00048 | TERMINATED | 10.164.33.209:1997117 |           64 |  0.149733 | 3.18804e-05 |            6 | 0.840502    |  0.671429  |                    4 |
| train_model_a34bb_00049 | TERMINATED | 10.164.33.209:2000562 |           16 |  0.41735  | 0.000464337 |            6 | 1.14337     |  0.578571  |                    2 |
| train_model_a34bb_00050 | TERMINATED | 10.164.33.209:2002351 |           64 |  0.153579 | 3.87286e-05 |            3 | 1.91628     |  0.178571  |                    1 |
| train_model_a34bb_00051 | TERMINATED | 10.164.33.209:2003281 |           64 |  0.156646 | 0.000391904 |            3 | 1.76997     |  0.307143  |                    1 |
| train_model_a34bb_00052 | TERMINATED | 10.164.33.209:2004207 |           32 |  0.175197 | 1.11086e-05 |            3 | 1.90909     |  0.185714  |                    1 |
| train_model_a34bb_00053 | TERMINATED | 10.164.33.209:2005138 |           16 |  0.125658 | 0.000777871 |            6 | 0.00116143  |  0.8       |                   30 |
| train_model_a34bb_00054 | TERMINATED | 10.164.33.209:2030803 |           32 |  0.128708 | 0.000179107 |            4 | 0.594514    |  0.678571  |                    4 |
| train_model_a34bb_00055 | TERMINATED | 10.164.33.209:2034263 |           64 |  0.274718 | 6.29762e-05 |            3 | 1.91018     |  0.192857  |                    1 |
| train_model_a34bb_00056 | TERMINATED | 10.164.33.209:2035192 |           64 |  0.168391 | 4.97148e-05 |            4 | 1.8478      |  0.292857  |                    1 |
| train_model_a34bb_00057 | TERMINATED | 10.164.33.209:2036120 |           32 |  0.28392  | 0.000351029 |            4 | 0.490692    |  0.657143  |                    4 |
| train_model_a34bb_00058 | TERMINATED | 10.164.33.209:2039582 |           64 |  0.212836 | 1.45518e-05 |            4 | 1.89947     |  0.185714  |                    1 |
| train_model_a34bb_00059 | TERMINATED | 10.164.33.209:2040515 |           32 |  0.442348 | 0.000666915 |            4 | 1.07446     |  0.585714  |                    2 |
| train_model_a34bb_00060 | TERMINATED | 10.164.33.209:2042290 |           32 |  0.201627 | 0.00030214  |            4 | 0.550077    |  0.685714  |                    4 |
| train_model_a34bb_00061 | TERMINATED | 10.164.33.209:2045751 |           64 |  0.230338 | 0.000869757 |            4 | 0.321188    |  0.714286  |                    8 |
| train_model_a34bb_00062 | TERMINATED | 10.164.33.209:2052544 |           32 |  0.351767 | 8.44732e-05 |            5 | 1.60126     |  0.364286  |                    1 |
| train_model_a34bb_00063 | TERMINATED | 10.164.33.209:2053476 |           64 |  0.108756 | 0.000446895 |            6 | 1.57443     |  0.371429  |                    1 |
| train_model_a34bb_00064 | TERMINATED | 10.164.33.209:2054404 |           32 |  0.438515 | 0.000206447 |            4 | 1.44955     |  0.507143  |                    2 |
| train_model_a34bb_00065 | TERMINATED | 10.164.33.209:2056182 |           16 |  0.450625 | 0.000193256 |            6 | 1.53134     |  0.228571  |                    1 |
| train_model_a34bb_00066 | TERMINATED | 10.164.33.209:2057122 |           16 |  0.355446 | 5.1715e-05  |            4 | 1.77811     |  0.214286  |                    1 |
| train_model_a34bb_00067 | TERMINATED | 10.164.33.209:2058061 |           16 |  0.17522  | 1.87577e-05 |            4 | 1.64445     |  0.535714  |                    2 |
| train_model_a34bb_00068 | TERMINATED | 10.164.33.209:2059848 |           16 |  0.169822 | 0.000815786 |            6 | 1.22948     |  0.514286  |                    2 |
| train_model_a34bb_00069 | TERMINATED | 10.164.33.209:2061641 |           64 |  0.420939 | 6.79001e-05 |            4 | 1.84803     |  0.207143  |                    1 |
| train_model_a34bb_00070 | TERMINATED | 10.164.33.209:2062576 |           32 |  0.214041 | 0.000118678 |            4 | 0.836403    |  0.671429  |                    4 |
| train_model_a34bb_00071 | TERMINATED | 10.164.33.209:2066034 |           16 |  0.334004 | 0.00025996  |            4 | 0.589763    |  0.592857  |                    4 |
| train_model_a34bb_00072 | TERMINATED | 10.164.33.209:2069522 |           64 |  0.260093 | 0.00034849  |            6 | 1.87894     |  0.414286  |                    1 |
| train_model_a34bb_00073 | TERMINATED | 10.164.33.209:2070449 |           32 |  0.210704 | 0.000460664 |            6 | 0.470731    |  0.628571  |                    4 |
| train_model_a34bb_00074 | TERMINATED | 10.164.33.209:2073908 |           32 |  0.24157  | 1.18984e-05 |            6 | 1.60925     |  0.257143  |                    1 |
| train_model_a34bb_00075 | TERMINATED | 10.164.33.209:2074838 |           32 |  0.422153 | 3.03998e-05 |            3 | 1.90711     |  0.307143  |                    1 |
| train_model_a34bb_00076 | TERMINATED | 10.164.33.209:2075769 |           64 |  0.315925 | 7.91082e-05 |            3 | 1.88191     |  0.2       |                    1 |
| train_model_a34bb_00077 | TERMINATED | 10.164.33.209:2076695 |           16 |  0.233034 | 0.000237026 |            3 | 0.479958    |  0.728571  |                    8 |
| train_model_a34bb_00078 | TERMINATED | 10.164.33.209:2083587 |           64 |  0.463521 | 0.000569574 |            5 | 1.22728     |  0.528571  |                    2 |
| train_model_a34bb_00079 | TERMINATED | 10.164.33.209:2085353 |           32 |  0.263494 | 0.000346218 |            5 | 0.00959679  |  0.835714  |                   30 |
| train_model_a34bb_00080 | TERMINATED | 10.164.33.209:2110715 |           32 |  0.257762 | 6.69686e-05 |            5 | 1.55334     |  0.25      |                    1 |
| train_model_a34bb_00081 | TERMINATED | 10.164.33.209:2111645 |           32 |  0.230948 | 2.4533e-05  |            5 | 1.69683     |  0.371429  |                    1 |
| train_model_a34bb_00082 | TERMINATED | 10.164.33.209:2112576 |           16 |  0.19974  | 0.000328893 |            5 | 0.161442    |  0.771429  |                   16 |
| train_model_a34bb_00083 | TERMINATED | 10.164.33.209:2126261 |           32 |  0.369912 | 1.78541e-05 |            5 | 1.88755     |  0.278571  |                    1 |
| train_model_a34bb_00084 | TERMINATED | 10.164.33.209:2127192 |           32 |  0.174146 | 0.000347349 |            6 | 1.0368      |  0.585714  |                    2 |
| train_model_a34bb_00085 | TERMINATED | 10.164.33.209:2128966 |           64 |  0.349613 | 0.000192531 |            6 | 1.13189     |  0.528571  |                    2 |
| train_model_a34bb_00086 | TERMINATED | 10.164.33.209:2130733 |           16 |  0.213078 | 0.000178782 |            6 | 0.441628    |  0.671429  |                    4 |
| train_model_a34bb_00087 | TERMINATED | 10.164.33.209:2134229 |           32 |  0.452979 | 9.10464e-05 |            6 | 1.59467     |  0.128571  |                    1 |
| train_model_a34bb_00088 | TERMINATED | 10.164.33.209:2135160 |           16 |  0.408205 | 1.04308e-05 |            5 | 1.87074     |  0.257143  |                    1 |
| train_model_a34bb_00089 | TERMINATED | 10.164.33.209:2136099 |           16 |  0.446527 | 3.08583e-05 |            3 | 1.91602     |  0.235714  |                    1 |
| train_model_a34bb_00090 | TERMINATED | 10.164.33.209:2137037 |           64 |  0.120759 | 0.000143882 |            4 | 1.71091     |  0.364286  |                    1 |
| train_model_a34bb_00091 | TERMINATED | 10.164.33.209:2137969 |           16 |  0.167773 | 0.00033872  |            5 | 0.0901589   |  0.828571  |                   30 |
| train_model_a34bb_00092 | TERMINATED | 10.164.33.209:2163589 |           16 |  0.244786 | 3.03219e-05 |            4 | 1.59688     |  0.442857  |                    2 |
| train_model_a34bb_00093 | TERMINATED | 10.164.33.209:2165376 |           64 |  0.154466 | 1.8194e-05  |            5 | 1.75698     |  0.407143  |                    1 |
| train_model_a34bb_00094 | TERMINATED | 10.164.33.209:2166304 |           64 |  0.429419 | 0.00053541  |            3 | 1.62686     |  0.492857  |                    2 |
| train_model_a34bb_00095 | TERMINATED | 10.164.33.209:2168072 |           16 |  0.478611 | 0.000641705 |            5 | 1.42135     |  0.407143  |                    1 |
| train_model_a34bb_00096 | TERMINATED | 10.164.33.209:2169013 |           16 |  0.386872 | 4.60223e-05 |            6 | 1.56843     |  0.0857143 |                    1 |
| train_model_a34bb_00097 | TERMINATED | 10.164.33.209:2169961 |           16 |  0.407859 | 0.000427787 |            5 | 0.267532    |  0.757143  |                    8 |
| train_model_a34bb_00098 | TERMINATED | 10.164.33.209:2176858 |           16 |  0.466663 | 9.43433e-05 |            3 | 1.83194     |  0.392857  |                    1 |
| train_model_a34bb_00099 | TERMINATED | 10.164.33.209:2177797 |           64 |  0.354982 | 0.000449242 |            5 | 0.138033    |  0.764286  |                   16 |
| train_model_a34bb_00100 | TERMINATED | 10.164.33.209:2191297 |           32 |  0.352299 | 1.1321e-05  |            5 | 1.82078     |  0.35      |                    1 |
| train_model_a34bb_00101 | TERMINATED | 10.164.33.209:2192236 |           16 |  0.467616 | 0.000306387 |            3 | 1.44076     |  0.521429  |                    2 |
| train_model_a34bb_00102 | TERMINATED | 10.164.33.209:2194023 |           32 |  0.23854  | 5.89759e-05 |            3 | 1.92317     |  0.3       |                    1 |
| train_model_a34bb_00103 | TERMINATED | 10.164.33.209:2194956 |           32 |  0.136641 | 0.000136514 |            3 | 1.62558     |  0.535714  |                    2 |
| train_model_a34bb_00104 | TERMINATED | 10.164.33.209:2196729 |           32 |  0.309953 | 0.000354555 |            4 | 0.341895    |  0.685714  |                    8 |
| train_model_a34bb_00105 | TERMINATED | 10.164.33.209:2203555 |           16 |  0.43673  | 0.000107019 |            4 | 1.73897     |  0.285714  |                    1 |
| train_model_a34bb_00106 | TERMINATED | 10.164.33.209:2204495 |           64 |  0.195994 | 0.000172929 |            4 | 1.71179     |  0.257143  |                    1 |
| train_model_a34bb_00107 | TERMINATED | 10.164.33.209:2205421 |           32 |  0.386019 | 0.000160628 |            6 | 1.59274     |  0.271429  |                    1 |
| train_model_a34bb_00108 | TERMINATED | 10.164.33.209:2206354 |           64 |  0.360085 | 1.25349e-05 |            4 | 1.98773     |  0.107143  |                    1 |
| train_model_a34bb_00109 | TERMINATED | 10.164.33.209:2207280 |           16 |  0.391775 | 5.60302e-05 |            6 | 1.49355     |  0.107143  |                    1 |
| train_model_a34bb_00110 | TERMINATED | 10.164.33.209:2208218 |           16 |  0.223948 | 0.000184162 |            4 | 0.314139    |  0.75      |                    8 |
| train_model_a34bb_00111 | TERMINATED | 10.164.33.209:2215118 |           64 |  0.455107 | 0.000182541 |            6 | 1.61278     |  0.385714  |                    1 |
| train_model_a34bb_00112 | TERMINATED | 10.164.33.209:2216046 |           16 |  0.218807 | 0.000166125 |            4 | 0.327046    |  0.707143  |                    8 |
| train_model_a34bb_00113 | TERMINATED | 10.164.33.209:2222939 |           64 |  0.439701 | 8.39441e-05 |            6 | 1.72739     |  0.2       |                    1 |
| train_model_a34bb_00114 | TERMINATED | 10.164.33.209:2223870 |           16 |  0.477875 | 0.000317758 |            3 | 1.44351     |  0.557143  |                    2 |
| train_model_a34bb_00115 | TERMINATED | 10.164.33.209:2225657 |           64 |  0.378912 | 0.000249439 |            3 | 1.86481     |  0.278571  |                    1 |
| train_model_a34bb_00116 | TERMINATED | 10.164.33.209:2226586 |           32 |  0.316013 | 0.00032576  |            4 | 1.09166     |  0.542857  |                    2 |
| train_model_a34bb_00117 | TERMINATED | 10.164.33.209:2228359 |           16 |  0.489591 | 0.000679656 |            4 | 1.11524     |  0.55      |                    2 |
| train_model_a34bb_00118 | TERMINATED | 10.164.33.209:2230149 |           32 |  0.428327 | 0.000454246 |            6 | 1.32385     |  0.457143  |                    2 |
| train_model_a34bb_00119 | TERMINATED | 10.164.33.209:2231924 |           64 |  0.186241 | 0.00018108  |            6 | 0.304058    |  0.75      |                    8 |
| train_model_a34bb_00120 | TERMINATED | 10.164.33.209:2238726 |           32 |  0.203465 | 5.2716e-05  |            5 | 0.507202    |  0.7       |                    8 |
| train_model_a34bb_00121 | TERMINATED | 10.164.33.209:2245548 |           64 |  0.352496 | 4.95723e-05 |            3 | 1.92236     |  0.257143  |                    1 |
| train_model_a34bb_00122 | TERMINATED | 10.164.33.209:2246477 |           16 |  0.392844 | 0.000568453 |            5 | 0.45877     |  0.635714  |                    4 |
| train_model_a34bb_00123 | TERMINATED | 10.164.33.209:2249969 |           32 |  0.194113 | 1.44976e-05 |            4 | 1.89893     |  0.192857  |                    1 |
| train_model_a34bb_00124 | TERMINATED | 10.164.33.209:2250900 |           64 |  0.461959 | 0.000287793 |            4 | 1.78095     |  0.292857  |                    1 |
| train_model_a34bb_00125 | TERMINATED | 10.164.33.209:2251830 |           64 |  0.226082 | 4.51585e-05 |            3 | 1.95194     |  0.185714  |                    1 |
| train_model_a34bb_00126 | TERMINATED | 10.164.33.209:2252759 |           16 |  0.221106 | 5.97811e-05 |            5 | 0.353169    |  0.757143  |                    8 |
| train_model_a34bb_00127 | TERMINATED | 10.164.33.209:2259650 |           64 |  0.462856 | 3.69628e-05 |            6 | 1.79873     |  0.164286  |                    1 |
| train_model_a34bb_00128 | TERMINATED | 10.164.33.209:2260578 |           64 |  0.306247 | 1.44039e-05 |            4 | 1.89901     |  0.307143  |                    1 |
| train_model_a34bb_00129 | TERMINATED | 10.164.33.209:2261505 |           32 |  0.410335 | 3.37018e-05 |            4 | 1.86178     |  0.264286  |                    1 |
| train_model_a34bb_00130 | TERMINATED | 10.164.33.209:2262435 |           16 |  0.103859 | 1.8924e-05  |            3 | 1.89089     |  0.314286  |                    1 |
| train_model_a34bb_00131 | TERMINATED | 10.164.33.209:2263375 |           64 |  0.216926 | 0.000258563 |            4 | 1.66048     |  0.4       |                    1 |
| train_model_a34bb_00132 | TERMINATED | 10.164.33.209:2264300 |           16 |  0.486278 | 0.00052487  |            6 | 1.3324      |  0.535714  |                    2 |
| train_model_a34bb_00133 | TERMINATED | 10.164.33.209:2266093 |           32 |  0.227854 | 0.000257968 |            4 | 0.657404    |  0.65      |                    4 |
| train_model_a34bb_00134 | TERMINATED | 10.164.33.209:2269553 |           16 |  0.492272 | 5.72695e-05 |            4 | 1.78504     |  0.164286  |                    1 |
| train_model_a34bb_00135 | TERMINATED | 10.164.33.209:2270492 |           64 |  0.280372 | 3.37356e-05 |            5 | 1.75188     |  0.314286  |                    1 |
| train_model_a34bb_00136 | TERMINATED | 10.164.33.209:2271428 |           16 |  0.351281 | 1.68357e-05 |            6 | 1.62887     |  0.0857143 |                    1 |
| train_model_a34bb_00137 | TERMINATED | 10.164.33.209:2272368 |           32 |  0.329188 | 0.000124491 |            5 | 1.53486     |  0.357143  |                    1 |
| train_model_a34bb_00138 | TERMINATED | 10.164.33.209:2273304 |           16 |  0.203934 | 2.80461e-05 |            3 | 1.92802     |  0.4       |                    1 |
| train_model_a34bb_00139 | TERMINATED | 10.164.33.209:2274244 |           64 |  0.18978  | 0.000221742 |            5 | 1.09044     |  0.564286  |                    2 |
| train_model_a34bb_00140 | TERMINATED | 10.164.33.209:2276010 |           32 |  0.267383 | 3.77715e-05 |            4 | 1.83265     |  0.207143  |                    1 |
| train_model_a34bb_00141 | TERMINATED | 10.164.33.209:2276940 |           16 |  0.391906 | 0.000318187 |            5 | 1.4629      |  0.392857  |                    1 |
| train_model_a34bb_00142 | TERMINATED | 10.164.33.209:2277878 |           16 |  0.407881 | 0.000228936 |            5 | 1.43685     |  0.335714  |                    1 |
| train_model_a34bb_00143 | TERMINATED | 10.164.33.209:2278816 |           16 |  0.258659 | 0.000112335 |            6 | 0.0972682   |  0.807143  |                   30 |
| train_model_a34bb_00144 | TERMINATED | 10.164.33.209:2304435 |           32 |  0.147355 | 0.00030607  |            5 | 0.533949    |  0.707143  |                    4 |
| train_model_a34bb_00145 | TERMINATED | 10.164.33.209:2307905 |           64 |  0.485543 | 0.000426236 |            4 | 1.72032     |  0.285714  |                    1 |
| train_model_a34bb_00146 | TERMINATED | 10.164.33.209:2308834 |           64 |  0.185109 | 5.24168e-05 |            6 | 1.23753     |  0.564286  |                    2 |
| train_model_a34bb_00147 | TERMINATED | 10.164.33.209:2310601 |           16 |  0.255178 | 5.57208e-05 |            3 | 1.71393     |  0.471429  |                    2 |
| train_model_a34bb_00148 | TERMINATED | 10.164.33.209:2312393 |           64 |  0.131937 | 0.000801025 |            6 | 2.03531     |  0.314286  |                    1 |
| train_model_a34bb_00149 | TERMINATED | 10.164.33.209:2313322 |           32 |  0.366169 | 0.000769322 |            5 | 0.683246    |  0.664286  |                    4 |
| train_model_a34bb_00150 | TERMINATED | 10.164.33.209:2316786 |           32 |  0.490033 | 2.87555e-05 |            3 | 1.91168     |  0.228571  |                    1 |
| train_model_a34bb_00151 | TERMINATED | 10.164.33.209:2317721 |           16 |  0.40314  | 2.54607e-05 |            5 | 1.74341     |  0.171429  |                    1 |
| train_model_a34bb_00152 | TERMINATED | 10.164.33.209:2318662 |           16 |  0.128597 | 0.000151553 |            4 | 0.860438    |  0.585714  |                    2 |
| train_model_a34bb_00153 | TERMINATED | 10.164.33.209:2320452 |           16 |  0.40778  | 3.48569e-05 |            3 | 1.91945     |  0.242857  |                    1 |
| train_model_a34bb_00154 | TERMINATED | 10.164.33.209:2321391 |           16 |  0.331166 | 9.27385e-05 |            6 | 1.51808     |  0.292857  |                    1 |
| train_model_a34bb_00155 | TERMINATED | 10.164.33.209:2322330 |           64 |  0.114856 | 0.000591362 |            5 | 1.27342     |  0.335714  |                    1 |
| train_model_a34bb_00156 | TERMINATED | 10.164.33.209:2323257 |           32 |  0.488118 | 3.09685e-05 |            6 | 1.7398      |  0.0857143 |                    1 |
| train_model_a34bb_00157 | TERMINATED | 10.164.33.209:2324188 |           32 |  0.248186 | 0.000743532 |            6 | 1.22465     |  0.478571  |                    2 |
| train_model_a34bb_00158 | TERMINATED | 10.164.33.209:2325961 |           16 |  0.256199 | 0.000580734 |            4 | 0.344397    |  0.728571  |                    8 |
| train_model_a34bb_00159 | TERMINATED | 10.164.33.209:2332847 |           16 |  0.392994 | 0.000551595 |            3 | 1.1939      |  0.564286  |                    2 |
| train_model_a34bb_00160 | TERMINATED | 10.164.33.209:2334636 |           64 |  0.194311 | 9.67506e-05 |            4 | 1.77427     |  0.335714  |                    1 |
| train_model_a34bb_00161 | TERMINATED | 10.164.33.209:2335564 |           16 |  0.25804  | 2.23416e-05 |            4 | 1.80785     |  0.378571  |                    1 |
| train_model_a34bb_00162 | TERMINATED | 10.164.33.209:2336501 |           64 |  0.280636 | 0.000833822 |            4 | 0.220957    |  0.785714  |                   16 |
| train_model_a34bb_00163 | TERMINATED | 10.164.33.209:2349996 |           64 |  0.113142 | 1.0593e-05  |            5 | 1.88188     |  0.335714  |                    1 |
| train_model_a34bb_00164 | TERMINATED | 10.164.33.209:2350923 |           32 |  0.473872 | 9.6049e-05  |            3 | 1.88857     |  0.292857  |                    1 |
| train_model_a34bb_00165 | TERMINATED | 10.164.33.209:2351855 |           32 |  0.430753 | 4.9198e-05  |            6 | 1.60471     |  0.114286  |                    1 |
| train_model_a34bb_00166 | TERMINATED | 10.164.33.209:2352786 |           64 |  0.230754 | 1.10965e-05 |            4 | 1.95893     |  0.257143  |                    1 |
| train_model_a34bb_00167 | TERMINATED | 10.164.33.209:2353714 |           32 |  0.473848 | 1.12399e-05 |            6 | 1.93293     |  0.0857143 |                    1 |
| train_model_a34bb_00168 | TERMINATED | 10.164.33.209:2354644 |           64 |  0.115306 | 1.14036e-05 |            5 | 1.87924     |  0.307143  |                    1 |
| train_model_a34bb_00169 | TERMINATED | 10.164.33.209:2355574 |           64 |  0.140044 | 5.733e-05   |            3 | 1.91764     |  0.278571  |                    1 |
| train_model_a34bb_00170 | TERMINATED | 10.164.33.209:2356501 |           32 |  0.494708 | 3.36272e-05 |            3 | 1.92294     |  0.257143  |                    1 |
| train_model_a34bb_00171 | TERMINATED | 10.164.33.209:2357437 |           32 |  0.348012 | 0.000560207 |            5 | 0.247918    |  0.757143  |                    8 |
| train_model_a34bb_00172 | TERMINATED | 10.164.33.209:2364258 |           16 |  0.191193 | 0.000912476 |            6 | 1.12933     |  0.4       |                    2 |
| train_model_a34bb_00173 | TERMINATED | 10.164.33.209:2366049 |           16 |  0.172802 | 0.000511684 |            3 | 1.0024      |  0.571429  |                    2 |
| train_model_a34bb_00174 | TERMINATED | 10.164.33.209:2367845 |           16 |  0.284194 | 0.000621151 |            6 | 0.649031    |  0.628571  |                    4 |
| train_model_a34bb_00175 | TERMINATED | 10.164.33.209:2371337 |           64 |  0.374563 | 3.02235e-05 |            4 | 1.89701     |  0.192857  |                    1 |
| train_model_a34bb_00176 | TERMINATED | 10.164.33.209:2372264 |           32 |  0.183209 | 1.19845e-05 |            5 | 1.59236     |  0.464286  |                    2 |
| train_model_a34bb_00177 | TERMINATED | 10.164.33.209:2374045 |           16 |  0.290144 | 0.000321844 |            4 | 0.354654    |  0.678571  |                    8 |
| train_model_a34bb_00178 | TERMINATED | 10.164.33.209:2380933 |           16 |  0.382306 | 0.000300132 |            3 | 0.759939    |  0.55      |                    4 |
| train_model_a34bb_00179 | TERMINATED | 10.164.33.209:2384437 |           16 |  0.393179 | 1.6321e-05  |            3 | 1.95785     |  0.235714  |                    1 |
| train_model_a34bb_00180 | TERMINATED | 10.164.33.209:2385375 |           32 |  0.473224 | 1.77518e-05 |            6 | 1.84986     |  0.0857143 |                    1 |
| train_model_a34bb_00181 | TERMINATED | 10.164.33.209:2386307 |           64 |  0.220502 | 8.50004e-05 |            3 | 1.81794     |  0.25      |                    2 |
| train_model_a34bb_00182 | TERMINATED | 10.164.33.209:2388071 |           32 |  0.49325  | 0.000336298 |            4 | 1.36131     |  0.585714  |                    2 |
| train_model_a34bb_00183 | TERMINATED | 10.164.33.209:2389843 |           16 |  0.11175  | 0.00011494  |            4 | 0.68191     |  0.692857  |                    4 |
| train_model_a34bb_00184 | TERMINATED | 10.164.33.209:2393333 |           64 |  0.14116  | 1.42735e-05 |            4 | 1.89824     |  0.171429  |                    1 |
| train_model_a34bb_00185 | TERMINATED | 10.164.33.209:2394259 |           64 |  0.103242 | 8.19639e-05 |            3 | 1.93204     |  0.185714  |                    1 |
| train_model_a34bb_00186 | TERMINATED | 10.164.33.209:2395188 |           32 |  0.356984 | 0.000815181 |            5 | 1.63557     |  0.321429  |                    1 |
| train_model_a34bb_00187 | TERMINATED | 10.164.33.209:2396119 |           32 |  0.479052 | 2.08704e-05 |            4 | 1.94538     |  0.278571  |                    1 |
| train_model_a34bb_00188 | TERMINATED | 10.164.33.209:2397049 |           32 |  0.361303 | 1.44595e-05 |            4 | 1.92502     |  0.228571  |                    1 |
| train_model_a34bb_00189 | TERMINATED | 10.164.33.209:2397979 |           16 |  0.389242 | 0.000402655 |            4 | 0.954595    |  0.571429  |                    2 |
| train_model_a34bb_00190 | TERMINATED | 10.164.33.209:2399769 |           32 |  0.361174 | 0.000375394 |            3 | 1.01739     |  0.614286  |                    4 |
| train_model_a34bb_00191 | TERMINATED | 10.164.33.209:2403225 |           64 |  0.350436 | 8.92107e-05 |            3 | 1.93968     |  0.214286  |                    1 |
| train_model_a34bb_00192 | TERMINATED | 10.164.33.209:2404151 |           16 |  0.186143 | 1.23621e-05 |            6 | 1.50784     |  0.378571  |                    1 |
| train_model_a34bb_00193 | TERMINATED | 10.164.33.209:2405090 |           16 |  0.370058 | 0.000873585 |            3 | 0.57039     |  0.7       |                    4 |
| train_model_a34bb_00194 | TERMINATED | 10.164.33.209:2408577 |           16 |  0.171981 | 0.000121321 |            5 | 0.153027    |  0.785714  |                   16 |
| train_model_a34bb_00195 | TERMINATED | 10.164.33.209:2422287 |           32 |  0.198887 | 0.000519394 |            3 | 1.26258     |  0.585714  |                    2 |
| train_model_a34bb_00196 | TERMINATED | 10.164.33.209:2424060 |           16 |  0.375008 | 4.85031e-05 |            3 | 1.87653     |  0.378571  |                    1 |
| train_model_a34bb_00197 | TERMINATED | 10.164.33.209:2425000 |           32 |  0.316981 | 0.000296382 |            5 | 0.599646    |  0.7       |                    4 |
| train_model_a34bb_00198 | TERMINATED | 10.164.33.209:2428455 |           64 |  0.146878 | 1.84822e-05 |            5 | 1.81311     |  0.385714  |                    1 |
| train_model_a34bb_00199 | TERMINATED | 10.164.33.209:2429382 |           64 |  0.224107 | 2.50321e-05 |            5 | 1.93872     |  0.328571  |                    1 |
| train_model_a34bb_00200 | TERMINATED | 10.164.33.209:2430309 |           64 |  0.260228 | 0.000815029 |            4 | 0.670585    |  0.678571  |                    4 |
| train_model_a34bb_00201 | TERMINATED | 10.164.33.209:2433756 |           16 |  0.253437 | 1.07711e-05 |            5 | 1.71477     |  0.371429  |                    1 |
| train_model_a34bb_00202 | TERMINATED | 10.164.33.209:2434697 |           32 |  0.302855 | 0.000184135 |            4 | 1.29322     |  0.578571  |                    2 |
| train_model_a34bb_00203 | TERMINATED | 10.164.33.209:2436469 |           32 |  0.340182 | 0.000117448 |            6 | 1.4746      |  0.278571  |                    1 |
| train_model_a34bb_00204 | TERMINATED | 10.164.33.209:2437401 |           64 |  0.323269 | 0.000862018 |            6 | 1.9123      |  0.357143  |                    1 |
| train_model_a34bb_00205 | TERMINATED | 10.164.33.209:2438329 |           16 |  0.132852 | 0.000221039 |            6 | 0.000611483 |  0.814286  |                   30 |
| train_model_a34bb_00206 | TERMINATED | 10.164.33.209:2463948 |           16 |  0.182914 | 0.000960612 |            5 | 0.101239    |  0.707143  |                   16 |
| train_model_a34bb_00207 | TERMINATED | 10.164.33.209:2477662 |           32 |  0.160888 | 0.000297887 |            4 | 1.00491     |  0.585714  |                    2 |
| train_model_a34bb_00208 | TERMINATED | 10.164.33.209:2479435 |           32 |  0.305826 | 7.37546e-05 |            3 | 1.84926     |  0.264286  |                    1 |
| train_model_a34bb_00209 | TERMINATED | 10.164.33.209:2480370 |           16 |  0.260746 | 0.000375135 |            4 | 0.545249    |  0.621429  |                    4 |
| train_model_a34bb_00210 | TERMINATED | 10.164.33.209:2483866 |           64 |  0.427786 | 0.000234795 |            4 | 1.77903     |  0.214286  |                    1 |
| train_model_a34bb_00211 | TERMINATED | 10.164.33.209:2484794 |           64 |  0.248551 | 0.000794042 |            3 | 1.40167     |  0.557143  |                    2 |
| train_model_a34bb_00212 | TERMINATED | 10.164.33.209:2486560 |           16 |  0.190088 | 1.06831e-05 |            5 | 1.48719     |  0.55      |                    2 |
| train_model_a34bb_00213 | TERMINATED | 10.164.33.209:2488348 |           32 |  0.144927 | 0.000526396 |            5 | 0.84475     |  0.514286  |                    2 |
| train_model_a34bb_00214 | TERMINATED | 10.164.33.209:2490123 |           64 |  0.493569 | 0.000535703 |            5 | 1.72828     |  0.328571  |                    1 |
| train_model_a34bb_00215 | TERMINATED | 10.164.33.209:2491052 |           32 |  0.323542 | 2.05332e-05 |            3 | 1.93739     |  0.221429  |                    1 |
| train_model_a34bb_00216 | TERMINATED | 10.164.33.209:2491984 |           16 |  0.201704 | 1.799e-05   |            5 | 1.70603     |  0.257143  |                    1 |
| train_model_a34bb_00217 | TERMINATED | 10.164.33.209:2492924 |           32 |  0.286581 | 0.000161054 |            6 | 0.665771    |  0.592857  |                    4 |
| train_model_a34bb_00218 | TERMINATED | 10.164.33.209:2496388 |           32 |  0.48688  | 3.50685e-05 |            3 | 1.88449     |  0.3       |                    1 |
| train_model_a34bb_00219 | TERMINATED | 10.164.33.209:2497317 |           32 |  0.407207 | 0.000231233 |            6 | 1.659       |  0.378571  |                    1 |
| train_model_a34bb_00220 | TERMINATED | 10.164.33.209:2498251 |           32 |  0.350475 | 0.000383208 |            4 | 1.16918     |  0.557143  |                    2 |
| train_model_a34bb_00221 | TERMINATED | 10.164.33.209:2500030 |           64 |  0.246308 | 0.000152861 |            5 | 1.21104     |  0.585714  |                    2 |
| train_model_a34bb_00222 | TERMINATED | 10.164.33.209:2501795 |           16 |  0.377662 | 2.86761e-05 |            3 | 1.91822     |  0.228571  |                    1 |
| train_model_a34bb_00223 | TERMINATED | 10.164.33.209:2502734 |           32 |  0.278285 | 0.000116059 |            4 | 1.41708     |  0.557143  |                    2 |
| train_model_a34bb_00224 | TERMINATED | 10.164.33.209:2504514 |           64 |  0.299391 | 4.77285e-05 |            4 | 1.85275     |  0.292857  |                    1 |
| train_model_a34bb_00225 | TERMINATED | 10.164.33.209:2505444 |           16 |  0.385433 | 2.32556e-05 |            3 | 1.93096     |  0.292857  |                    1 |
| train_model_a34bb_00226 | TERMINATED | 10.164.33.209:2506385 |           64 |  0.27638  | 2.02226e-05 |            6 | 1.76122     |  0.371429  |                    1 |
| train_model_a34bb_00227 | TERMINATED | 10.164.33.209:2507315 |           32 |  0.399467 | 0.000444883 |            5 | 0.902974    |  0.578571  |                    2 |
| train_model_a34bb_00228 | TERMINATED | 10.164.33.209:2509094 |           32 |  0.19765  | 0.000193994 |            3 | 1.56346     |  0.514286  |                    2 |
| train_model_a34bb_00229 | TERMINATED | 10.164.33.209:2510873 |           32 |  0.185466 | 0.00015187  |            3 | 1.62102     |  0.557143  |                    2 |
| train_model_a34bb_00230 | TERMINATED | 10.164.33.209:2512647 |           16 |  0.341169 | 7.15188e-05 |            6 | 1.45463     |  0.228571  |                    1 |
| train_model_a34bb_00231 | TERMINATED | 10.164.33.209:2513586 |           16 |  0.232977 | 0.000401234 |            6 | 0.000575801 |  0.821429  |                   30 |
| train_model_a34bb_00232 | TERMINATED | 10.164.33.209:2539223 |           16 |  0.135343 | 3.01732e-05 |            6 | 0.483501    |  0.671429  |                    4 |
| train_model_a34bb_00233 | TERMINATED | 10.164.33.209:2542721 |           16 |  0.127379 | 7.51124e-05 |            6 | 0.204592    |  0.75      |                    8 |
| train_model_a34bb_00234 | TERMINATED | 10.164.33.209:2549628 |           32 |  0.372633 | 0.000252215 |            3 | 1.62551     |  0.492857  |                    2 |
| train_model_a34bb_00235 | TERMINATED | 10.164.33.209:2551401 |           16 |  0.301066 | 2.21639e-05 |            6 | 1.50082     |  0.164286  |                    1 |
| train_model_a34bb_00236 | TERMINATED | 10.164.33.209:2552366 |           64 |  0.147243 | 0.000280629 |            5 | 1.53029     |  0.378571  |                    1 |
| train_model_a34bb_00237 | TERMINATED | 10.164.33.209:2553353 |           16 |  0.447046 | 8.18054e-05 |            4 | 1.79342     |  0.178571  |                    1 |
| train_model_a34bb_00238 | TERMINATED | 10.164.33.209:2554293 |           32 |  0.263623 | 4.01559e-05 |            3 | 1.90119     |  0.285714  |                    1 |
| train_model_a34bb_00239 | TERMINATED | 10.164.33.209:2555235 |           64 |  0.421805 | 0.000101737 |            5 | 1.74512     |  0.242857  |                    1 |
| train_model_a34bb_00240 | TERMINATED | 10.164.33.209:2556172 |           16 |  0.128049 | 5.87998e-05 |            3 | 1.21632     |  0.621429  |                    4 |
| train_model_a34bb_00241 | TERMINATED | 10.164.33.209:2559662 |           16 |  0.390322 | 4.71397e-05 |            3 | 1.87187     |  0.321429  |                    1 |
| train_model_a34bb_00242 | TERMINATED | 10.164.33.209:2560601 |           16 |  0.377545 | 0.000174226 |            4 | 1.25266     |  0.542857  |                    2 |
| train_model_a34bb_00243 | TERMINATED | 10.164.33.209:2562400 |           16 |  0.279625 | 1.13084e-05 |            3 | 1.91528     |  0.142857  |                    1 |
| train_model_a34bb_00244 | TERMINATED | 10.164.33.209:2563338 |           32 |  0.111284 | 0.000707875 |            4 | 0.816154    |  0.578571  |                    2 |
| train_model_a34bb_00245 | TERMINATED | 10.164.33.209:2565111 |           32 |  0.1455   | 2.24106e-05 |            5 | 0.446495    |  0.721429  |                    8 |
| train_model_a34bb_00246 | TERMINATED | 10.164.33.209:2571950 |           32 |  0.452241 | 0.000211079 |            5 | 0.302295    |  0.742857  |                    8 |
| train_model_a34bb_00247 | TERMINATED | 10.164.33.209:2578771 |           32 |  0.292593 | 0.000772972 |            4 | 0.50903     |  0.642857  |                    4 |
| train_model_a34bb_00248 | TERMINATED | 10.164.33.209:2582229 |           16 |  0.248435 | 2.4628e-05  |            4 | 1.81399     |  0.25      |                    1 |
| train_model_a34bb_00249 | TERMINATED | 10.164.33.209:2583173 |           32 |  0.428241 | 0.000229762 |            3 | 1.66159     |  0.485714  |                    2 |
| train_model_a34bb_00250 | TERMINATED | 10.164.33.209:2584947 |           64 |  0.353787 | 3.31001e-05 |            5 | 1.76247     |  0.307143  |                    1 |
| train_model_a34bb_00251 | TERMINATED | 10.164.33.209:2585876 |           32 |  0.294653 | 0.00014864  |            5 | 0.0345401   |  0.807143  |                   30 |
| train_model_a34bb_00252 | TERMINATED | 10.164.33.209:2611226 |           64 |  0.223945 | 0.000183484 |            4 | 1.7113      |  0.342857  |                    1 |
| train_model_a34bb_00253 | TERMINATED | 10.164.33.209:2612160 |           16 |  0.477285 | 9.50231e-05 |            6 | 1.59049     |  0.1       |                    1 |
| train_model_a34bb_00254 | TERMINATED | 10.164.33.209:2613101 |           64 |  0.292456 | 2.50695e-05 |            6 | 1.67672     |  0.364286  |                    1 |
| train_model_a34bb_00255 | TERMINATED | 10.164.33.209:2614030 |           64 |  0.450133 | 1.42858e-05 |            5 | 1.91855     |  0.278571  |                    1 |
| train_model_a34bb_00256 | TERMINATED | 10.164.33.209:2614958 |           32 |  0.481066 | 0.000277347 |            4 | 1.47276     |  0.557143  |                    2 |
| train_model_a34bb_00257 | TERMINATED | 10.164.33.209:2616738 |           16 |  0.334813 | 0.000705261 |            5 | 0.925251    |  0.585714  |                    2 |
| train_model_a34bb_00258 | TERMINATED | 10.164.33.209:2618526 |           64 |  0.149571 | 0.000924631 |            3 | 1.35615     |  0.578571  |                    2 |
| train_model_a34bb_00259 | TERMINATED | 10.164.33.209:2620297 |           64 |  0.194703 | 7.91675e-05 |            4 | 1.81987     |  0.235714  |                    1 |
| train_model_a34bb_00260 | TERMINATED | 10.164.33.209:2621225 |           64 |  0.213656 | 2.98906e-05 |            4 | 1.8843      |  0.242857  |                    1 |
| train_model_a34bb_00261 | TERMINATED | 10.164.33.209:2622152 |           32 |  0.315787 | 7.16212e-05 |            5 | 1.58775     |  0.271429  |                    1 |
| train_model_a34bb_00262 | TERMINATED | 10.164.33.209:2623091 |           64 |  0.122506 | 0.00021436  |            3 | 1.84263     |  0.3       |                    1 |
| train_model_a34bb_00263 | TERMINATED | 10.164.33.209:2624018 |           64 |  0.170339 | 0.000205746 |            6 | 0.116211    |  0.785714  |                   16 |
| train_model_a34bb_00264 | TERMINATED | 10.164.33.209:2637534 |           16 |  0.373275 | 4.91633e-05 |            5 | 1.59919     |  0.235714  |                    1 |
| train_model_a34bb_00265 | TERMINATED | 10.164.33.209:2638473 |           64 |  0.342933 | 0.000126602 |            4 | 1.78896     |  0.3       |                    1 |
| train_model_a34bb_00266 | TERMINATED | 10.164.33.209:2639401 |           32 |  0.498792 | 0.00096824  |            5 | 1.13496     |  0.521429  |                    2 |
| train_model_a34bb_00267 | TERMINATED | 10.164.33.209:2641175 |           32 |  0.194042 | 0.00013149  |            6 | 0.184794    |  0.764286  |                   16 |
| train_model_a34bb_00268 | TERMINATED | 10.164.33.209:2654750 |           32 |  0.136982 | 0.000165667 |            3 | 1.53369     |  0.557143  |                    2 |
| train_model_a34bb_00269 | TERMINATED | 10.164.33.209:2656523 |           32 |  0.36069  | 2.33112e-05 |            6 | 1.63489     |  0.1       |                    1 |
| train_model_a34bb_00270 | TERMINATED | 10.164.33.209:2657459 |           64 |  0.389569 | 0.000608494 |            3 | 1.8181      |  0.214286  |                    1 |
| train_model_a34bb_00271 | TERMINATED | 10.164.33.209:2658390 |           16 |  0.499288 | 0.000537764 |            6 | 1.57716     |  0.371429  |                    1 |
| train_model_a34bb_00272 | TERMINATED | 10.164.33.209:2659330 |           16 |  0.351337 | 6.67976e-05 |            3 | 1.87532     |  0.264286  |                    1 |
| train_model_a34bb_00273 | TERMINATED | 10.164.33.209:2660268 |           64 |  0.444049 | 0.000164478 |            5 | 1.35966     |  0.542857  |                    2 |
| train_model_a34bb_00274 | TERMINATED | 10.164.33.209:2662036 |           16 |  0.487849 | 1.44444e-05 |            4 | 1.87299     |  0.257143  |                    1 |
| train_model_a34bb_00275 | TERMINATED | 10.164.33.209:2662977 |           32 |  0.318472 | 8.13287e-05 |            5 | 0.0700018   |  0.814286  |                   30 |
| train_model_a34bb_00276 | TERMINATED | 10.164.33.209:2688362 |           64 |  0.372665 | 0.000206409 |            5 | 1.24374     |  0.542857  |                    2 |
| train_model_a34bb_00277 | TERMINATED | 10.164.33.209:2690132 |           32 |  0.489489 | 4.23022e-05 |            6 | 1.67523     |  0.0857143 |                    1 |
| train_model_a34bb_00278 | TERMINATED | 10.164.33.209:2691063 |           32 |  0.434808 | 0.000278535 |            4 | 1.33552     |  0.578571  |                    2 |
| train_model_a34bb_00279 | TERMINATED | 10.164.33.209:2692837 |           64 |  0.329697 | 1.11878e-05 |            4 | 1.97512     |  0.185714  |                    1 |
| train_model_a34bb_00280 | TERMINATED | 10.164.33.209:2693767 |           64 |  0.331082 | 0.000177632 |            6 | 1.08981     |  0.578571  |                    2 |
| train_model_a34bb_00281 | TERMINATED | 10.164.33.209:2695539 |           32 |  0.144765 | 1.57796e-05 |            5 | 1.71258     |  0.357143  |                    1 |
| train_model_a34bb_00282 | TERMINATED | 10.164.33.209:2696471 |           16 |  0.443646 | 2.47386e-05 |            3 | 1.90847     |  0.264286  |                    1 |
| train_model_a34bb_00283 | TERMINATED | 10.164.33.209:2697410 |           16 |  0.412962 | 0.000381749 |            3 | 0.718472    |  0.642857  |                    4 |
| train_model_a34bb_00284 | TERMINATED | 10.164.33.209:2700900 |           64 |  0.106008 | 0.00036507  |            5 | 0.9379      |  0.585714  |                    2 |
| train_model_a34bb_00285 | TERMINATED | 10.164.33.209:2702664 |           64 |  0.356319 | 1.74119e-05 |            3 | 1.94743     |  0.171429  |                    1 |
| train_model_a34bb_00286 | TERMINATED | 10.164.33.209:2703593 |           64 |  0.264833 | 7.2535e-05  |            6 | 1.26247     |  0.507143  |                    2 |
| train_model_a34bb_00287 | TERMINATED | 10.164.33.209:2705366 |           64 |  0.475787 | 9.14632e-05 |            4 | 1.823       |  0.214286  |                    1 |
| train_model_a34bb_00288 | TERMINATED | 10.164.33.209:2706294 |           16 |  0.187672 | 0.000132291 |            4 | 0.568098    |  0.635714  |                    4 |
| train_model_a34bb_00289 | TERMINATED | 10.164.33.209:2709786 |           16 |  0.103247 | 0.000404977 |            4 | 0.368538    |  0.714286  |                    8 |
| train_model_a34bb_00290 | TERMINATED | 10.164.33.209:2716689 |           64 |  0.108826 | 4.53379e-05 |            5 | 1.47207     |  0.478571  |                    2 |
| train_model_a34bb_00291 | TERMINATED | 10.164.33.209:2718461 |           64 |  0.25799  | 0.000272014 |            6 | 1.70819     |  0.371429  |                    1 |
| train_model_a34bb_00292 | TERMINATED | 10.164.33.209:2719390 |           16 |  0.370698 | 0.000175239 |            4 | 1.30318     |  0.571429  |                    2 |
| train_model_a34bb_00293 | TERMINATED | 10.164.33.209:2721179 |           32 |  0.400416 | 0.000113935 |            5 | 1.56245     |  0.214286  |                    1 |
| train_model_a34bb_00294 | TERMINATED | 10.164.33.209:2722113 |           16 |  0.446923 | 0.000752122 |            6 | 1.30995     |  0.578571  |                    2 |
| train_model_a34bb_00295 | TERMINATED | 10.164.33.209:2723902 |           64 |  0.220448 | 0.000568644 |            5 | 1.39471     |  0.378571  |                    1 |
| train_model_a34bb_00296 | TERMINATED | 10.164.33.209:2724829 |           32 |  0.405136 | 7.17743e-05 |            4 | 1.80408     |  0.357143  |                    1 |
| train_model_a34bb_00297 | TERMINATED | 10.164.33.209:2725767 |           64 |  0.241011 | 1.18505e-05 |            4 | 1.91573     |  0.185714  |                    1 |
| train_model_a34bb_00298 | TERMINATED | 10.164.33.209:2726694 |           16 |  0.341724 | 2.33805e-05 |            6 | 1.55257     |  0.107143  |                    1 |
| train_model_a34bb_00299 | TERMINATED | 10.164.33.209:2727635 |           32 |  0.237098 | 2.29929e-05 |            5 | 1.48727     |  0.578571  |                    2 |
+-------------------------+------------+-----------------------+--------------+-----------+-------------+--------------+-------------+------------+----------------------+


Best trial config: {'num_classes': 7, 'lr': 0.000346217591697255, 'batch_size': 32, 'num_layers': 5, 'dropout': 0.2634941123610691}
Best trial final validation loss: 0.009596792744773933
Best trial final validation accuracy: 0.8357142857142857
    '''