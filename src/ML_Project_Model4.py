#@title ***Importing required librarys***

import numpy as np
from sklearn import svm
from ML_Project_Utilities import read_images, run_model_with_cv

#@title ***Model 4***

# Reading images & labels
images, labels = read_images()

# Flatting images
images = np.array([img.flatten() for img in images])

# model classifier
model = svm.SVC(C=10, gamma=0.001)

# Running model
run_model_with_cv('model4', model, images, labels)