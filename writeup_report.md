#Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a weights for a convolutional neural network
* model.json containing a description of a convolutional neural network
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.json
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model. The code is mostly self-explanatory but it includes some comments to clarify one or two lines.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the [Nvidia research paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) which was recommended for use with the project. My keras implementation can be seen in lines 25-50 of model.py.

The model includes Exponential Linear Unit (ELU) layers to introduce nonlinearity instead of ReLU units. This is because ELU units are supposed to promote faster learning ([e.g.](http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/)).

Data is normalized in the model using a Keras lambda layer (code line 28). Each image is also cropped in line 31, in an attempt to remove irrelevant pixels (e.g. sky, car) from the image and to focus primarily on the road and road boundaries.

####2. Attempts to reduce overfitting in the model

The model contains dropout (50%) layers in order to reduce overfitting (model.py lines 42-48). The dropout is applied only to the fully-connected layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model.py "train" method is invoked in main.py on line 24. The actual splitting into training and validation data occurs in training_data.py on line 61; 90% of the data is used for training with the remaining 10% used for validation. The entire dataset is shuffled before splitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. It occasionally runs over the painted lines but doesn't mount the ledge or drive off the paved surface at any point.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 20).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the one proposed by Nvidia in [their recent paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) I thought this model might be appropriate because the course materials told me so! From the course material, comments from my assigned mentor, chatter on the forums, etc. it seemed evident that this would be a good starting point and I was strongly urged to use it and to not consider other means. I had originally intended to adapt some of the models described in the coursework (e.g. VGG) but moved to the Nvidia model based on these endorsements. For reasons I won't get into in this write-up, I ended up not having time to run the other models as well to compare them to the Nvidia one...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to add Dropout layers to all the fully-connected layers. This seemed to do the trick.

The only other modification to the Nvidia model was to change the output to produce a single linear output that corresponds to a (normalised) steering angle for the simulator.

The final step was to run the simulator to see how well the car was driving around track one. The car initially drove erratically and after some iterations of the code (adding dropout and cropping the image) it drove mostly straight but was still unable to navigate major turns. I spent a lot of time generating additional data (and, frankly, wasting time fighting the simulator trying to get usable data; I'm not a talented video gamer and the learning curve to produce data without leaving the road was significant) to improve the driving behavior in these cases. Initially no matter how much data I used the model didn't improve. After some debugging I realised I had a bug in my code whereby my training data was 90 examples rather than 90% of the data! 

After fixing the bug I had amassed quite a lot of training data and so the model improved enormously. It still struggled a little bit at points that weren't frequently seen in the data, e.g. the bridge, the dirt track, etc. It also struggled to recover from the edge of the road back to the centre. I spent some time recording myself driving back and forth across the bridge and past the dirt track, then did a few laps of veering to the edge of the road and then recording myself returning to the centre. I think some of the training data I created this way is not very good due to my aforementioned ineptitude with the simulator, but the majority of it works well. There are just occasionally some very pronounced turns that are a bit more aggressive than they need to be!

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture, as mentioned above, is the same basic architecture as the NVidia end-to-end driving model, with the addition of dropout layers and a single output.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then supplanted it with many more laps in both directions. That proved to not quite be suitable so I recorded some additional data showing the vehicle recovering from the left side and right sides of the road back to center. The idea was that the model would learn that when it sees a road boundary close to the centre of the image or when it sees the boundary directly in front of the car it should turn sharply away from the boundary. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

I ended up getting good enough results from track 1 and didn't train with track 2 at all.

To augment the data sat, I also flipped images and angles thinking that this would double the amount of training data and prevent the model from having a bias for turning in any one direction. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had ~ 180,000 data points (doubled to 360,000 with random flipping). I didn't perform any pre-processing of the images outside the model. Instead the model normalizes each pixel to a value between-1 and 1 and crops the non-road sections of the image (based on hard-coded parameters; this approach would not generalise to other terrain types).

I finally randomly shuffled the data set and put 10% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by a plateau in learning improvements. I used an adam optimizer so that manually training the learning rate wasn't necessary.
