import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def upload_CTFSG(token, grader, file):
    import urllib.request, os, json
    urllib.request.urlretrieve('https://raw.githubusercontent.com/alttablabs/ctfsg-utils/master/pyctfsglib.py', './pyctfsglib.py')
    print('Downloaded pyctfsglib.py:', 'pyctfsglib.py' in os.listdir())
    import pyctfsglib as ctfsg
    grader = ctfsg.DSGraderClient(grader, token)
    response = json.loads(grader.submitFile(file))
    os.rename(file, f'{response["multiplier"]}_sklearn_{file[:-4]}.csv')
    return response

#Init Train
train = pd.read_csv('../train.csv')
audio_files = train['file'].values
audio_files = ['../sounds/' + i for i in audio_files]
labels = train['symbol'].values
le = LabelEncoder()
labels = le.fit_transform(labels)

#Init Test
test = pd.read_csv('../test.csv')
test_audio_files = test['file'].values
test_audio_files = ['../sounds/' + i for i in test_audio_files]

def extract_features(audio_file, n_mfcc=20, n_chroma=12, n_spectral_contrast=6):
    y, sr = librosa.load(audio_file)
    features = []
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features.extend(np.mean(mfcc.T, axis=0))
    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
    features.extend(np.mean(chroma.T, axis=0))
    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_spectral_contrast)
    features.extend(np.mean(spectral_contrast.T, axis=0))
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    features.extend(np.mean(tonnetz.T, axis=0))
    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    features.extend(np.mean(zero_crossing_rate.T, axis=0))
    # Spectral Roll-off
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    features.extend(np.mean(spectral_rolloff.T, axis=0))

    print(f'Extracted features from {audio_file}')
    return features

'''
#Model (0.8443)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
clf1 = ExtraTreesClassifier(criterion="gini", n_estimators=100, random_state=2023)
clf2 = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=2023, tol=1e-5))
from sklearn.ensemble import HistGradientBoostingClassifier
clf3 = HistGradientBoostingClassifier(random_state=2023)
clf4 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
from sklearn.linear_model import SGDClassifier
clf5 = make_pipeline(StandardScaler(), SGDClassifier(loss='modified_huber', tol=1e-5, random_state=2023))
eclf = VotingClassifier(estimators=[('et', clf1), ('svc', clf2), ('hgb', clf3), ('xgb', clf4), ('SGD',clf5)], voting='soft')
'''

# Base classifiers
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
base_classifiers = [
    ('et_gini', ExtraTreesClassifier(criterion='gini', n_estimators=100, random_state=2023)),
    ('linear_svc', make_pipeline(StandardScaler(), LinearSVC(random_state=2023))),
    ('hgb', HistGradientBoostingClassifier(random_state=2023)),
    ('sgd', make_pipeline(StandardScaler(), SGDClassifier(loss='modified_huber', tol=1e-5, random_state=2023))),
    ('et_entropy', ExtraTreesClassifier(criterion='entropy', n_estimators=100, random_state=2023)),
    ('ridge', make_pipeline(StandardScaler(), RidgeClassifier(random_state=2023))),
    ('nu_svc', make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=2023, tol=1e-5))),
    ('mlp', make_pipeline(StandardScaler(), MLPClassifier(random_state=2023))),
]

# Meta classifier
meta_classifier = LogisticRegression(random_state=2023)

# Stacking classifier
clf = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier, cv=5)

'''#Testing
#Train Model
features = [extract_features(audio_file) for audio_file in audio_files]
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2023)
eclf.fit(X_train, y_train)
y_pred = eclf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Ensemble accuracy: {accuracy:.2f}")
'''

#Train Model
#Honestly not the best to train on all the data me thinks but whatever...
features = [extract_features(audio_file) for audio_file in audio_files]
clf.fit(features, labels)

#Predict
test_features = [extract_features(audio_file) for audio_file in test_audio_files]
y_pred = clf.predict(test_features)
y_pred = le.inverse_transform(y_pred)
test['symbol'] = y_pred
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