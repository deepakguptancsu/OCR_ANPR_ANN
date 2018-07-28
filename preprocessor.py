import numpy as np 
import cv2
import pandas as pd 
import matplotlib.pyplot
import csv
import os

df = pd.DataFrame(columns=range(0,626),index=range(0,50000))

values = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

dic = {}
k = 0.0
for i in values:
	dic[i] = k
	k += 1.0

rootdir = "./training_data/all_train"

subdirs = [x[0] for x in os.walk(rootdir)]

i = 0
for dirs in subdirs:
	for filename in os.listdir(dirs):
		if filename[1] == '_':
			label = int(filename[0])
		else:
			label = int(filename[0:2])
		img = cv2.imread(rootdir + "/" + filename)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
		print img.shape
		img = img.reshape(1,625)
		val = np.array(label)
		arr = np.append(val,np.asfarray(img))
		print i, rootdir + "/" + filename, label
		df.iloc[i] = np.append(val,np.asfarray(img))
		i += 1
	break


print i
df.to_csv('full_train.csv',index=False)
