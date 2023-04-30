import cv2
import numpy as np
from scipy.stats import pearsonr

def upload_CTFSG(token, grader, file):
    import urllib.request, os, json
    urllib.request.urlretrieve('https://raw.githubusercontent.com/alttablabs/ctfsg-utils/master/pyctfsglib.py', './pyctfsglib.py')
    print('Downloaded pyctfsglib.py:', 'pyctfsglib.py' in os.listdir())
    import pyctfsglib as ctfsg
    grader = ctfsg.DSGraderClient(grader, token)
    response = json.loads(grader.submitFile(file))
    os.rename(file, f'{response["multiplier"]}_{file[:-4]}.csv')
    return response

def preprocess_image(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    edges = cv2.Canny(mask, 50, 200)
    return edges

def detect_points(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    points = []
    for contour in contours:
        if cv2.contourArea(contour) > 5:
            moments = cv2.moments(contour)
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
            points.append(center)
    return points

def draw_points(image, points):
    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 5, (0, 0, 255), 2)
    cv2.imshow("Detected Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_correlation(points):
    if len(points) < 2:
        print("Insufficient points for correlation calculation. At least 2 points are required.")
        return None

    x = [point[0] for point in points]
    y = [point[1] for point in points]
    correlation, _ = pearsonr(x,y)
    return correlation

def process(filename):
    image_path = filename
    image = cv2.imread(image_path)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Resize the image to zoom in
    thresholded_image = preprocess_image(image)
    points = detect_points(thresholded_image)
    draw_points(image.copy(), points)  # Display the original image with detected points
    correlation = calculate_correlation(points)
    
    return (correlation) * -1 #this is honestly a ??? moment

#open test folder and compute correlation for each image using process function.
#Then save it in a csv ordered by filename
import os
import csv
import pandas as pd
from datetime import datetime

path = 'test'
files = os.listdir(path)
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
outFile = f'submission_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
with open(outFile, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image", "correlation"])
    for i in range(0, len(files)):
        correlation= process(path + '/' + str(i) + '.jpg')
        writer.writerow([i, correlation])
        if i % 100 == 0:
            print(i)

import random
GRADER_URL = random.choice([
  "https://correlated01.ds.chals.t.cyberthon23.ctf.sg/",
  "https://correlated02.ds.chals.t.cyberthon23.ctf.sg/"
])
token = "NrMxsaIrKbsxNvHNoNbEnljIXTxsWLQYUtnVpHSzyQrqEIPYjXZglMuvDjomTEhd"
print(upload_CTFSG(token, GRADER_URL, outFile))




