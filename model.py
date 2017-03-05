from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.optimizers import Adam

'''
A class to encapsulate a Keras model based on the Nvidia end-to-end classifier.

Includes wrapper functions to handle definition, training, and saving to disk.
'''

class NvidiaModel:

    INPUT_IMG_HEIGHT = 160
    INPUT_IMG_WEIGHT = 320
    INPUT_IMG_DEPTH = 3 # colour images
    
    def __init__(self, loss_function="mse"):
        self.loss_function = loss_function
        self.optimizer = Adam(lr=0.001)
        self.define_model()


    def define_model(self):
        self.model = Sequential()

        #normalize inputs
        self.model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(self.INPUT_IMG_HEIGHT, self.INPUT_IMG_WEIGHT, self.INPUT_IMG_DEPTH)))

        #crop edges to remove irrelevant data (car, sky, etc)
        self.model.add(Cropping2D(cropping=((10,10),(60,30))))

        #NVidia model based on https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
        #with additional dropout functions after the fully-connected layers to prevent overfitting
        self.model.add(Convolution2D(24, 5, 5, border_mode="valid", subsample=(2, 2), activation='elu'))
        self.model.add(Convolution2D(36, 5, 5, border_mode="valid", subsample=(2, 2), activation='elu'))
        self.model.add(Convolution2D(48, 3, 3, border_mode="valid", subsample=(2, 2), activation='elu'))
        self.model.add(Convolution2D(64, 3, 3, border_mode="valid", subsample=(2, 2), activation='elu'))
        self.model.add(Convolution2D(64, 3, 3, border_mode="valid", subsample=(2, 2), activation='elu'))
        self.model.add(Flatten())
        self.model.add(Dense(1164, activation="elu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(100, activation="elu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(50, activation="elu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation="elu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation="linear"))
        self.model.compile(loss=self.loss_function, optimizer=self.optimizer)

    def train(self, training_data, validation_data, epochs, training_samples_per_epoch, validation_samples_per_epoch):
        self.model.fit_generator(training_data,
                                 samples_per_epoch=training_samples_per_epoch,
                                 nb_epoch=epochs,
                                 validation_data=validation_data,
                                 nb_val_samples=validation_samples_per_epoch)

    def save(self, weights_file="model.h5", model_file="model.json"):
        self.model.save_weights(weights_file)
        json_string = self.model.to_json()
        with open(model_file, "w") as f:
            f.write(json_string)
