import warnings
import csv
import sys
import os
import random
import numpy as np
import matplotlib.image as mpimg
import cv2

'''
A class to encapsulate reading training data for the Udacity self-driving-car course simulator.

The class expects to be pointed at a CSV file containing driving angles and paths to images
captured using the simulator. The angle and the file location are read into memory and are then
used to create training and validation generators to be used in training the model.

Of note is the absence of a test set. The main goal of the project is to be able to "safely" 
navigate a lap of the simulator. In my opinion that is a better testing methodology than running
a test set on the model so I didn't bother to include a test set (and sacrifice some data).
Obviously it would be trivial to add it if necessary...

'''
class TrainingData:
    PATH = "path"
    ANGLE = "angle"
    ANGLE_CORRECTION = 0.23 #determined through trial-and-error
    TRAINING_DATA_SIZE = 0.9

    def __init__(self, csv_filepath, center_img_index=0, left_img_index=1, right_img_index=2, angle_index=3):
        self.csv_filepath = csv_filepath
        self.center_img_index = center_img_index
        self.left_img_index = left_img_index
        self.right_img_index = right_img_index
        self.angle_index = angle_index        
        self.data = list()
        print("Reading training data from '" + self.csv_filepath + "'...")        
        self.load_data()
        print("Data loaded.")
        self.split_data()
        print("Training data count: " + str(self.num_training_samples()))
        print("Validation data count: " + str(self.num_validation_samples()))       

    def load_data(self):      
        processed_img_count = 0
        with open(self.csv_filepath, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if not os.path.isfile(row[self.center_img_index]):
                    warnings.warn("WARNING: skipping invalid file " + row[self.center_img_index])
                else:
                    angle = float(row[self.angle_index])
                    self.data.append({self.PATH:row[self.center_img_index], self.ANGLE: angle})
                    processed_img_count += 1
                    if os.path.isfile(row[self.left_img_index]) and os.path.isfile(row[self.right_img_index]):
                        data.append({self.PATH:row[self.left_img_index], self.ANGLE: left + self.ANGLE_CORRECTION})
                        data.append({self.PATH:row[self.right_img_index], self.ANGLE: right - self.ANGLE_CORRECTION})
                        processed_img_count += 2

        print(str(processed_img_count) + " imgs processed")

    def split_data(self):
        random.shuffle(self.data)
        self.training_data = self.data[:int(len(self.data) * self.TRAINING_DATA_SIZE)]
        self.validation_data = self.data[int(len(self.data) * self.TRAINING_DATA_SIZE):]

    def train_generator(self, batch_size):
        while 1:
            image_data = []
            angles = []

            for i in range(0, batch_size):
                #this randomization was originally intended to support more image augmentation
                #strictly speaking, it is no longer necessary... But randomly flipping the image
                #and randomly selecting an image means we don't always provide mirror pairs and
                #we don't have to pre-compute the images in advance. It could definitely be done,
                #but this works fine and it also allows for extending the training set size by simply
                # requesting more examples (although they will obviously include duplicates).
                # At this point, it's clear that this approach works, so all things considered I'm
                # leaving it in!
                exemplar = self.training_data[np.random.randint(len(self.training_data))]
                image = mpimg.imread(exemplar[self.PATH])
                angle = float(exemplar[self.ANGLE])
                
                ind_flip = np.random.randint(2)
                if ind_flip==0:
                    image = cv2.flip(image,1)
                    angle = -angle

                image_data.append(image)
                angles.append(angle)

            yield (np.array(image_data), np.array(angles))

    def validation_generator(self, batch_size):
        while 1:
            image_data = []
            angles = []
            
            for i in range(0, batch_size):
                exemplar = self.validation_data[np.random.randint(len(self.validation_data))]
                image_data.append(mpimg.imread(exemplar[self.PATH]))
                angles.append(float(exemplar[self.ANGLE]))
                
            yield (np.array(image_data), np.array(angles))

    def num_training_samples(self):
        return len(self.training_data)

    def num_validation_samples(self):
        return len(self.validation_data)
