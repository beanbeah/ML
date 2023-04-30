#Load all CSV Files from each model, combine them using voting, and save the results to a CSV file
import pandas as pd

def upload_CTFSG(token, grader, file):
    import urllib.request, os, json
    urllib.request.urlretrieve('https://raw.githubusercontent.com/alttablabs/ctfsg-utils/master/pyctfsglib.py', './pyctfsglib.py')
    print('Downloaded pyctfsglib.py:', 'pyctfsglib.py' in os.listdir())
    import pyctfsglib as ctfsg
    grader = ctfsg.DSGraderClient(grader, token)
    response = json.loads(grader.submitFile(file))
    os.rename(file, f'{response["multiplier"]}_{file[:-4]}.csv')
    return response

cnn = pd.read_csv('CNN/0.8914_cnn_submission_20230406234648.csv')
en = pd.read_csv('EfficientNet/0.8886_efficientnet.csv')
resnet = pd.read_csv('resnet/0.8614_resnet18_submission_20230404153345.csv')
sk = pd.read_csv('sklearn/0.8557_sklearn_submission_20230404102338.csv')
tf = pd.read_csv('TF_Autokeras/0.7757_tf.csv')

test = pd.read_csv('test.csv')
test_audio_files = test['file'].values

#Combine CSV files into one using voting
for file in test_audio_files:
    #Load predictions for each model, and take the most common prediction
    cnn_pred = cnn.loc[cnn['file'] == file, 'symbol'].values[0]
    en_pred = en.loc[en['file'] == file, 'symbol'].values[0]
    resnet_pred = resnet.loc[resnet['file'] == file, 'symbol'].values[0]
    sk_pred = sk.loc[sk['file'] == file, 'symbol'].values[0]
    tf_pred = tf.loc[tf['file'] == file, 'symbol'].values[0]

    #Give more weight to the models with higher accuracy

    #Create a list of all predictions
    preds = [cnn_pred, cnn_pred, cnn_pred, en_pred, en_pred, en_pred, resnet_pred, resnet_pred]

    #Find the most common prediction
    pred = max(set(preds), key=preds.count)

    #Update the CSV file with the most common prediction
    test.loc[test['file'] == file, 'symbol'] = pred

#Save the CSV file
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