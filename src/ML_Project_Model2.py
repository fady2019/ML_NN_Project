#@title ***Importing required librarys***

from tensorflow import keras
from ML_Project_Utilities import read_images, run_nn_model

#@title ***Model 2***

# Reading images & labels
images, labels = read_images(True)

# Setting model up
model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=(100, 100)),
  keras.layers.Dense(10000, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])

# Running model
run_nn_model('model2', model, images, labels)