# Importing required librarys
from ML_Project_Utilities import read_images, run_nn_model
from tensorflow import keras


# Reading images & labels
images, labels = read_images()

# Setting model up
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

run_nn_model('model3', model, images, labels)





