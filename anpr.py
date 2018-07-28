from preprocess import preProcess
import os
import sys
from ann_lib import *
inputFile = sys.argv[1]
preProcess(inputFile)

imglist = []
for img in os.listdir("./tmp"):
	imglist.append(img)

imglist.sort()

#call the function to train a model and validate
#this call return the plate number as string
ann_call_fn(imglist)

for fileName in os.listdir("./tmp"):
    os.remove("./tmp/"+fileName)
