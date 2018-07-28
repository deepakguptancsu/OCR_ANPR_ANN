import cv2
import numpy
import heapq
import sys
from scipy.signal import argrelextrema
from scipy.signal import correlate2d
from scipy.stats import pearsonr
from scipy import ndimage
from skimage import measure
from skimage import segmentation
import imutils
import warnings
from matplotlib import cm
from skimage.feature import hog
import math
from PIL import Image
warnings.filterwarnings("ignore")

#Function to Localize plate from image
def localize(image):
	pic = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
	pic2=pic
	sobelx=cv2.Sobel(pic2, cv2.CV_8U, 1, 0, ksize=3)
	sobel=sobelx
	py = numpy.empty([sobel.shape[0],1])
	for i in range(0,sobel.shape[0]):
		py[i]=sum(sobel[i,])
	
	py = ndimage.gaussian_filter1d(py, 5)
	local_maxima = argrelextrema(py, numpy.greater,order=10)
	local_maxima=local_maxima[0]
	max = heapq.nlargest(4, range(len(py[local_maxima])), py[local_maxima].take)
	max = local_maxima[max]
	b0 = numpy.ndarray(4,dtype=numpy.int64)
	b1 = numpy.ndarray(4,dtype=numpy.int64)
	count = 0
	j=0
	for i in max:
		for j in range(i,-1,-1):
			if py[j] <= 0.55*py[i]:
				b0[count]=j
				break
		for j in range(i,py.shape[0]):
			if py[j] <= 0.55*py[i]:
				b1[count]=j
				break
		py[b0[count]:b1[count]+1]=0
		count = count + 1
	
	idealplate = cv2.imread("idealPlate.jpeg",cv2.IMREAD_GRAYSCALE)
	project=numpy.ndarray(idealplate.shape[1])
	for i in range(0,project.shape[0]):
		project[i]=sum(idealplate[:,i])
		
	max = -sys.maxsize
	index = 0
	width = pic.shape[1]

	for i in range(0,4):
		if b0[i]<b1[i]:
			picproject = numpy.ndarray(width)
			for j in range(0,width):
				picproject[i]=sum(pic[:,i])
			diff=abs(275-width)
			diffheight = abs(19-(b1[i]-b0[i]))
			if(width < 275):
				picpadded=numpy.pad(pic[b0[i]:b1[i],],((0,0),(0,diff)),'constant')
			else:
				picpadded=pic[b0[i]:b1[i],0:275]
			
			if((b1[i]-b0[i])<19):
				picpadded=numpy.pad(picpadded,((0,diffheight),(0,0)),'constant')
			else:
				picpadded=picpadded[0:19,:]
			corr=numpy.ndarray(275)
			for j in range(0,275):
				corr[j] = pearsonr(idealplate[:,j],picpadded[:,j])[0]
			corr=numpy.nan_to_num(corr)
			if max <= abs(numpy.amax(corr)):
				max = abs(numpy.amax(corr))
				index=i
	
	plateband = pic[b0[index]:b1[index],]
	
	#Horizontal Segmentation
	pic_equiv=cv2.equalizeHist(plateband)
	kernel = numpy.ones((5,5),numpy.uint8)
	pic_equiv2=cv2.morphologyEx(pic_equiv, cv2.MORPH_OPEN, kernel)
	pic_mod=pic_equiv-pic_equiv2
	ret,pic_mod=cv2.threshold(pic_mod,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	sobely=cv2.Sobel(pic_mod, cv2.CV_8U, 1, 0, ksize=3)
	dilation = cv2.dilate(sobely,kernel,iterations = 1)
	dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
	dilation = cv2.erode(dilation,kernel,iterations = 1)
	
	labels = measure.label(dilation, neighbors=8, background=0)
	mask = numpy.zeros(dilation.shape, dtype="uint8")
	
	# loop over the unique components
	for label in numpy.unique(labels):
		# if this is the background label, ignore it
		if label == 0:
			continue
	
		# otherwise, construct the label mask and count the
		# number of pixels 
		labelMask = numpy.zeros(dilation.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
	
		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
		if numPixels > 300:
			lmask=labelMask
			mask = cv2.add(mask, labelMask)
	
	py = numpy.empty([mask.shape[1],1])
	for i in range(0,mask.shape[1]):
		py[i]=sum(mask[:,i])
	
	for i in range(0,py.shape[0]):
		if (py[i]!= 0):
			leftmax = i
			break
			
	for i in range(py.shape[0]-1,0,-1):
		if (py[i] != 0):
			rightmax = i
			break
	
	plate = plateband[:,leftmax:rightmax]
	return plate

#Function to extract segmentated characters from plate
def segmentation(plateImage):
	segmentedImages = []
	segImgCoordinates = []
	
	resizedPlateImage = cv2.resize(plateImage, (140,33))
	resizedPlateImage=cv2.equalizeHist(resizedPlateImage)
	_,threshPlateImage=cv2.threshold(resizedPlateImage, 60, 255, cv2.THRESH_BINARY_INV)
	_,contour,_=cv2.findContours(threshPlateImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	
	for index,contourElement in enumerate(contour):
		if (validateContourElement(contourElement)):
			countourRectangle = cv2.minAreaRect(contourElement)
			countourBox = cv2.boxPoints(countourRectangle)
			segImgCoordinates.append(countourBox[0][0])
			centre=countourRectangle[0]
			cropped=cv2.getRectSubPix(threshPlateImage,(int(min(countourRectangle[1])), int(numpy.amax(countourRectangle[1]))) , centre)
			width,height=cropped.shape
			cropped = numpy.invert(cropped)
			croppedImg = Image.fromarray(cropped)
			croppedImgWidth, croppedImgHeight = croppedImg.size
			background = Image.new('RGBA', (40, 40), (255, 255, 255, 255))
			backgroundImgWidth, backgroundImgHeight = background.size
			offset = ((backgroundImgWidth - croppedImgWidth)/2, (backgroundImgHeight - croppedImgHeight)/2)
			offset = tuple(map(int,offset))
			background.paste(croppedImg, offset)
			background = numpy.array(background)
			background = cv2.cvtColor(background,cv2.COLOR_BGR2GRAY);
			segmentedImages.append(background)
	sortSegImgCoordinates = numpy.argsort(segImgCoordinates)
	
	for i in range(len(segmentedImages)):
		imgName = "./tmp/" + str(i) + ".jpg"
		cv2.imwrite(imgName, segmentedImages[sortSegImgCoordinates[i]])
		
		
#Function to Validate Contour Elements
def validateContourElement(contourElement):
    countourRectangle=cv2.minAreaRect(contourElement)  
    countourBox=cv2.boxPoints(countourRectangle)
    width=countourRectangle[1][0]
    height=countourRectangle[1][1]
    retVal=False
    if ((width!=0) & (height!=0)):
        if (((height/width>0.2) & (height>width)) | ((width/height>0.2) & (width>height))):
            if((height*width<1700) & (height*width>60)):
                retVal=True
    return retVal
	
	
def preProcess(image):
	plate = localize(image)
	segmentation(plate)
