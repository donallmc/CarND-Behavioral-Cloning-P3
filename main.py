import sys
from training_data import TrainingData
from model import NvidiaModel

'''
Usage: python main.py [path_to_driving_log]

Loads training data then creates a model and trains it to drive the Udacity self-driving car course simulator.

To run the model in the simulator, run the drive.py and simulator program included in the course.
'''

#modify these to affect model training
batch_size = 256
training_samples_per_epoch = 409600
num_epochs = 10
validation_samples_per_epoch = 20480

csv_filepath = sys.argv[1]
training_data = TrainingData(csv_filepath)
model = NvidiaModel()

print("Training model...")
model.train(training_data.train_generator(batch_size),
            training_data.validation_generator(batch_size),
            num_epochs,
            training_samples_per_epoch,
            validation_samples_per_epoch)

print("Model trained. Saving to disk...")
model.save()
print("Saved.")
