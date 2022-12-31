#@title ***Importing required librarys***

import numpy as np
import cv2 as cv
import glob
from tensorflow import keras
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import joblib

#@title ***Project path***

project_path = "."

#@title ***Reading images utility***

def read_images(gray_scale=False):
  print('Reading images...')
  images = []
  labels = []

  for label in range(0, 10):
    path = f"{project_path}/Dataset/{label}/*.JPG"
      
    for img_path in glob.glob(path):
      # reading image
      img = cv.imread(img_path)
      # resize image 
      img = cv.resize(img, (100, 100))
      
      if gray_scale:
        # convert image from BGR to gary
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      else:
        # convert image from BGR to RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)       
        
      images.append(img)
      labels.append(label)

  images = np.array(images)
  labels = np.array(labels)
    
    
  # normalize images
  if not gray_scale:
    # calculate average
    avg = np.average(images, axis=0)
    # subtract averages from each image.
    images = images - avg
      
  # divide each image by 255
  images = images / 255.0 

  print('Images read successfully.\n')

  return (images, labels)

#@title ***Get model measurements utility***

def get_measurements(true_y, pred_y, average='micro'):
  return {
      "accuracy": accuracy_score(true_y, pred_y),
      "recall": recall_score(true_y, pred_y, average=average),
      "precision": precision_score(true_y, pred_y, average=average),
      "fscore": f1_score(true_y, pred_y, average=average),
  }

#@title ***Running model with cross-validation utility***

def run_model_with_cv(model_id, clf, images, labels, train_size=0.8, k=5):
  # Spliting data into training & testing
  training_images, testing_images, training_labels, testing_labels =  train_test_split(images, labels, train_size=train_size)

  scoring=('accuracy', 'recall_micro', 'precision_micro', 'f1_micro')

  cv = cross_validate(clf, training_images, training_labels, scoring=scoring, verbose=k, cv=k, return_train_score=True, return_estimator=True)

  best_model_accuracy, best_model_idx  = 0, -1

  with open(f"{project_path}/{model_id}.txt", 'w') as file:
    file.write(f"Model: {clf}\n\n\n")

    file.write("Training\n")
    file.write("-------------\n\n")
    for key in cv.keys():
      if key == 'estimator':
        continue
      
      file.write(f"{key}: {cv[key]}\n\n")

    file.write("\n\n")

    file.write("Testing\n")
    file.write("-------------\n\n")
    for (idx, model) in enumerate(cv['estimator']):
      # making prediction
      predicted_labels = model.predict(testing_images)

      measurements = get_measurements(testing_labels, predicted_labels)

      if measurements['accuracy'] > best_model_accuracy:
        best_model_accuracy = measurements['accuracy']
        best_model_idx = idx

      # reporting model
      file.write(f"model#{idx+1}\n")
      file.write(f"accuracy: {measurements['accuracy']}\n")
      file.write(f"recall: {measurements['recall']}\n")
      file.write(f"precision: {measurements['precision']}\n")
      file.write(f"fscore: {measurements['fscore']}\n\n")

  # Saving model
  joblib.dump(cv['estimator'][best_model_idx], f"{project_path}/{model_id}.pkl")

#@title ***Running NN model utility***

def run_nn_model(model_id, model: keras.models.Sequential, images, labels, train_size=0.8, epochs=30, k=5):
  # Reporting model summary
  with open(f"{project_path}/{model_id}.txt", "w") as file:
    model.summary(print_fn=lambda x: file.write(x + '\n'))
    file.write('\n\n')

  # Spliting data into training & testing
  training_images, testing_images, training_labels, testing_labels =  train_test_split(images, labels, train_size=train_size)

  # Configuring model
  model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
  )

  best_model_acc = 0

  for i, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=k).split(training_images, training_labels)):
    print("\n***********************************")
    print(f"*************** k#{i+1} ***************")
    print("***********************************\n")

    # Fitting model
    model.fit(
      training_images[train_idx], 
      training_labels[train_idx], 
      epochs=epochs, 
      verbose=1,
      validation_data=(training_images[val_idx], training_labels[val_idx]),
    )

    # Evaluating model
    predicted_labels = model.predict(testing_images, verbose=1)
    measurements = get_measurements(testing_labels, np.argmax(predicted_labels, axis=1))

    # Saving model
    if measurements['accuracy'] > best_model_acc:
      best_model_acc = measurements['accuracy']
      model.save(f"{project_path}/{model_id}.h5")

    # Reporting model
    with open(f"{project_path}/{model_id}.txt", "a") as file: 
      file.write(f"k#{i+1}\n")
      file.write(f"accuracy = {measurements['accuracy']}\n")
      file.write(f"recall = {measurements['recall']}\n")
      file.write(f"precision = {measurements['precision']}\n")
      file.write(f"fscore = {measurements['fscore']}\n\n")

#@title ***Model 1***

# Reading images & labels
images, labels = read_images(True)

# Setting model up
model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=(100, 100)),
  keras.layers.Dense(1024, activation='relu'),
  keras.layers.Dense(512, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])

# Running model
run_nn_model('model1', model, images, labels)

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

#@title ***Model 3***

# Reading images & labels
images, labels = read_images()

# Setting model up
model = keras.models.Sequential([
  keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
  keras.layers.MaxPool2D((2, 2)),
  keras.layers.Conv2D(64, (3, 3), activation='relu'),
  keras.layers.MaxPool2D((2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(1024, activation='relu'),
  keras.layers.Dense(512, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])

# Running model
run_nn_model('model3', model, images, labels)

#@title ***Model 4***

# Reading images & labels
images, labels = read_images()

# Flatting images
images = np.array([img.flatten() for img in images])

# model classifier
model = svm.SVC(C=10, gamma=0.001)

# Running model
run_model_with_cv('model4', model, images, labels)