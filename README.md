[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## Deep Learning Project ##

In this project I used a fully-convolutional neural network to paint all pixels in an image which is part of a person. Two types of persons are identified, the “hero” target person, and everyone else. Identifying a specific person from everyone else are useful for “follow-me” operations of a drone. This is the final project for Udacity's Robotics Nanodegree Term 1. 

# Here is my writeup
- https://github.com/mithi/follow-me/blob/master/WRITEUP.pdf

# The original repository can be found here
- git clone https://github.com/udacity/RoboND-DeepLearning.git

# Here is the Jupyter Notebook 
- I ran this on Udacity Robotics Laboratory Community AMI in AWS 
- https://github.com/mithi/follow-me/blob/master/code/model_training.ipynb

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

**Dependencies**

You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit).

If for some reason you choose not to use Anaconda, you must install the following frameworks and packages on your system:
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

The data directory is organized as follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models

data/raw_sim_data/train/run1
data/raw_sim_data/validation/run1
```

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```
