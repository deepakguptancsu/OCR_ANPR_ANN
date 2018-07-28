from preprocess import preProcess
import os
import sys
import shutil

inputDir = sys.argv[1]

for fileName in os.listdir(inputDir):
   fileNameWithoutExtension = fileName[:len(fileName)-4]
   print(fileNameWithoutExtension) 
   os.mkdir("./" + inputDir + "/" + fileNameWithoutExtension)
   destinationPath = "./" + inputDir + "/" + fileNameWithoutExtension + "/"
   sourcePath = "./" + inputDir + "/" + fileName
   preProcess(fileName, inputDir)
   shutil.copy(sourcePath, destinationPath)