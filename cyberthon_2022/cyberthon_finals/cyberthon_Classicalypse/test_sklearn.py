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
train = pd.read_csv('train_set.csv')
labels = train['is_promoted'].values
#features is drop is_promoted and employee_id columns
features = train.drop(['is_promoted', 'employee_id'], axis=1)
#clean up data, remove NaN
train_features = features.fillna(0)

#Init Test
test_features = pd.read_csv('test_set.csv')
test_features = test_features.drop(['employee_id'], axis=1)
test_features = test_features.fillna(0)

'''
1 HistGradientBoosting 0.8086218158066623
2 ExtraTrees (Gini) 0.7896799477465709
3 ExtraTrees (Entropy) 0.7864141084258655
4 GradientBoosting 0.7864141084258655
5 Bagging 0.7798824297844547
'''

# Base classifiers (based on above results)
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
base_classifiers = [
    ('hgb', HistGradientBoostingClassifier(random_state=2023)),
    ('et_gini', ExtraTreesClassifier(criterion='gini', n_estimators=100, random_state=2023)),
    ('et_entropy', ExtraTreesClassifier(criterion='entropy', n_estimators=100, random_state=2023)),
    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)),
    ('bagging', BaggingClassifier(n_estimators=10, random_state=0))
]

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
# Meta classifier
meta_classifier = LogisticRegression(random_state=2023)
# Stacking classifier
clf = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier, cv=5)

#Train Model
#Honestly not the best to train on all the data me thinks but whatever...
clf.fit(train_features, labels)

#Predict
y_pred = clf.predict(test_features)
submission = pd.read_csv('submission.csv')
submission['is_promoted'] = y_pred
outFile = f'submission_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
submission.to_csv(outFile, index=False)

#Upload
import random
GRADER_URL = "http://chals.f.cyberthon23.ctf.sg:42031/"
token = "XQwqczVjRbNLIQbRNlsPvntYEeYqLuXwjWbhnLIKRpIJUjlfxsYmYglKFnFAeaOp"

print(upload_CTFSG(token, GRADER_URL, outFile))