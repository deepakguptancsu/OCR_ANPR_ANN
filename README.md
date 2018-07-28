Setup Steps:
This Environment need to be setup by using "Ubuntu 16.04 LTS Base" image. 


sudo apt-get install python2.7

sudo apt-get install python-pip

sudo python -m pip install numpy

sudo python2 -m pip install opencv-python

sudo python -m pip install scipy

sudo python -m pip install pandas

sudo python -m pip install tqdm

sudo python -m pip install hickle

sudo python -m pip install matplotlib

sudo python -m pip install imutils

sudo python -m pip install scikit-image


Before running the application download the training data file from the given URL - (https://drive.google.com/file/d/1dSHrk0BsC06fNIgI2PhMf02jyhgDqVlf/view?usp=sharing) and copy it to project folder, this file should be named as "full_train.csv".
For implementation of the project, we chose the data set from - http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz


To run the Application Use the following way to run it from cmd prompt:
python anpr.py <car image to predict the number plate>

Example: 
python anpr.py car1.jpg
