# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[image1]: ./report/center.jpg "Center Image"
[image2]: ./report/recover_1.jpg "Recover_1"
[image3]: ./report/recover_2.jpg "Recover_2"
[image4]: ./report/recover_3.jpg "Recover_3"
[image5]: ./report/flip.jpg "Flip"
[image6]: ./report/angles_hist.jpg "Angels Histogram"
[image7]: ./report/loss_graph.jpg "Loss Graph"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run2.mp4 a Video of the autonomous driving by my trained model
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes with depths between 24 and 48 (model.py lines 77-79) and 3x3 filter sizes with depths of 64 (lines 80-81 

The model includes RELU layers to introduce nonlinearity (code line 71-81), and the data is normalized in the model using a Keras lambda layer (code line 75). Furthermore, the input images are cropped using Keras Cropping2D layer (code line 76) 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets which contains over 90K images to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model doesn't contain any dropout layer because the model withou Dropout doesn't have overfitting and have a good performance on the simulator. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 88).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the nVidia slef-drive car network. I thought this model might be appropriate because the same network helped their car drive autonomously 98% of the testing time, which is to me fantastic. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. And to my surprise both the training loss and validation loss are almost the same, NO overfitting at all.  

The final step was to run the simulator to see how well the car was driving around track one. The vehicle is able to drive autonomously around the track without leaving the road, although it tends to drive to left while on the bridge. Fantastic !!!

#### 2. Final Model Architecture

The final model architecture (model.py lines 75-86) consisted of a convolution neural network with the following layers and layer sizes ...

|Layer                           | Shape    |
|--------------------------------|:--------:|
|Input                           | 160x320x3|
|Normalization                   |          | 
|Cropping                        |          |
|Convolution (valid, 5x5x24)     |          |
|Activation  (ReLU)              |          |
|Convolution (valid, 5x5x36)     |          |
|Activation  (ReLU)              |          |
|Convolution (valid, 5x5x48)     |          |
|Activation  (ReLU)              |          |
|Flatten                         |          |
|Dense                           | 100      |
|Dense                           | 50       |
|Dense                           | 10       |
|Dense                           | 1        |
 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover to the center of the lane when too much on the left or right side of the lane. These images show what a recovery looks like starting from right to left :

![alt text][image2]
![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image5]

After the collection process, I had 16784 number of data points. I then preprocessed this data so
that the final number of inputs for the network is 100704.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I created two generators (one for training and one for validation) two retrieve all the input images
without the need to read all the input images into memory at one time to avoid crashing my laptop.
Detail implementation can be found in code lines 21-55. 

Below is two graphics to show the distribution of the angles in the training dataset and the loss. 
![alt text][image6]
![alt text][image7]
