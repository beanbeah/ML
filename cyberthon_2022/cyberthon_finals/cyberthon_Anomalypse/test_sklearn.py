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
train = pd.read_csv('train.csv')
labels = train['label'].values
#features is drop is_promoted and employee_id columns
features = train.drop(['label'], axis=1)

#Init Test
test_features = pd.read_csv('test.csv')

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
clf = KNeighborsClassifier()


#Train Model
#Honestly not the best to train on all the data me thinks but whatever...
clf.fit(features, labels)

#Predict
y_pred = clf.predict(test_features)
submission = pd.read_csv('submission.csv')
submission['label'] = y_pred
outFile = f'submission_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
submission.to_csv(outFile, index=False)

#Upload
import random
GRADER_URL = "http://chals.f.cyberthon23.ctf.sg:42021/"
token = "XQwqczVjRbNLIQbRNlsPvntYEeYqLuXwjWbhnLIKRpIJUjlfxsYmYglKFnFAeaOp"

print(upload_CTFSG(token, GRADER_URL, outFile))