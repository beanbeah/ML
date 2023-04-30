from pebble import ProcessPool
from concurrent.futures import TimeoutError
from multiprocessing import freeze_support
import sys

import traceback
import timeit
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

def format_time(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

classifiers = {
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=0),
    "Bagging": BaggingClassifier(n_estimators=10, random_state=0),
    "ExtraTrees (Gini)": ExtraTreesClassifier(criterion="gini", n_estimators=100, random_state=0),
    "ExtraTrees (Entropy)": ExtraTreesClassifier(criterion="entropy", n_estimators=100, random_state=0),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
    "RandomForest": RandomForestClassifier(max_depth=2, random_state=0),
    "HistGradientBoosting": HistGradientBoostingClassifier(),
    "GaussianProcess": GaussianProcessClassifier(random_state=0),
    "PassiveAggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3),
    "Ridge": RidgeClassifier(),
    "SGDClassifier": make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)),
    "BernoulliNaiveBayes": BernoulliNB(),
    "CategoricalNaiveBayes": CategoricalNB(),
    "ComplementNaiveBayes": ComplementNB(),
    "GaussianNaiveBayes": GaussianNB(),
    "MultinomialNaiveBayes": MultinomialNB(),
    "KNeighbors": KNeighborsClassifier(n_neighbors=3),
    "RadiusNeighbors": RadiusNeighborsClassifier(radius=1.0),
    "NearestCentroid": NearestCentroid(),
    "MLP": MLPClassifier(random_state=1, max_iter=300),
    "LabelPropagation": LabelPropagation(),
    "LinearSVC": make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5)),
    "NuSVC": make_pipeline(StandardScaler(), NuSVC()),
    "SVC": make_pipeline(StandardScaler(), SVC(gamma='auto')),
    "DecisionTree": DecisionTreeClassifier(random_state=0),
    "ExtraTree": ExtraTreeClassifier(random_state=0)
}

def test_classifier(classifier_type, X_train, X_test, y_train, y_test): 
    try:
        start_time = timeit.default_timer()
        print("\nClassifier", classifier_type, "...")
        clf = classifiers[classifier_type]
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_predicted)
        print("\nClassifier", classifier_type, "accuracy is", accuracy)
        stop_time = timeit.default_timer()
        print("Classifier", classifier_type, "completed in", format_time(stop_time-start_time))
        return [classifier_type, accuracy]
    except Exception: 
        print("\nError for Classifier", classifier_type)
        traceback.print_exc()

def test_classifiers(X_train, X_test, y_train, y_test):
    result_queue = []
    with ProcessPool(max_workers=POOL) as pool:             
        multiple_results = [(pool.schedule(test_classifier, args=(key, X_train, X_test, y_train, y_test), timeout=TIMEOUT_SECONDS), key) for key in classifiers]
        for res in multiple_results:
            try:
                tmp = res[0].result()
                if tmp is not None:
                    result_queue.append(tmp)
            except TimeoutError:
                print("\nClassifier", res[1], "exceeded the time limit.")

    accuracy = {}
    for value in result_queue:
        accuracy[value[0]] = value[1]
    accuracy = {k: v for k, v in sorted(accuracy.items(), key=lambda item: item[1], reverse=True)}
    
    print("Results: \n")
    untested = set(classifiers.keys())
    i = 1
    for key in accuracy:
        print(i, key, accuracy[key])
        untested.remove(key)
        i += 1
    print("\nUntested Classifiers:", untested)

import librosa
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
    return features


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    freeze_support()
    print("Python version:", sys.version)
    print("Pebble version:", sys.modules['pebble'].__version__)
    print("Numpy version:", np.__version__)
    print("Pandas version:", pd.__version__)

    POOL = 8
    TIMEOUT_SECONDS = 120

    from sklearn.preprocessing import LabelEncoder
    train = pd.read_csv('../train.csv')
    audio_files = train['file'].values
    audio_files = ['../sounds/' + i for i in audio_files]
    labels = train['symbol'].values
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    features = [extract_features(audio_file) for audio_file in audio_files]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2023)

    test_classifiers(X_train, X_test, y_train, y_test)


'''
Results:

1 ExtraTrees (Gini) 0.8071428571428572
2 LinearSVC 0.8071428571428572
3 HistGradientBoosting 0.8
4 SGDClassifier 0.8
5 ExtraTrees (Entropy) 0.7928571428571428
6 Ridge 0.7857142857142857
7 NuSVC 0.7785714285714286
8 MLP 0.7714285714285715
9 SVC 0.7714285714285715
10 Bagging 0.7642857142857142
11 GradientBoosting 0.7357142857142858
12 KNeighbors 0.7071428571428572
13 PassiveAggressive 0.7
14 DecisionTree 0.6857142857142857
15 GaussianNaiveBayes 0.6642857142857143
16 RandomForest 0.5928571428571429
17 ExtraTree 0.5714285714285714
18 NearestCentroid 0.5285714285714286
19 BernoulliNaiveBayes 0.4928571428571429
20 AdaBoost 0.4
21 GaussianProcess 0.20714285714285716
22 LabelPropagation 0.09285714285714286

Untested Classifiers: {'ComplementNaiveBayes', 'CategoricalNaiveBayes', 'RadiusNeighbors', 'MultinomialNaiveBayes'}'''
