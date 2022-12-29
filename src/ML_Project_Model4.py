# Importing required librarys
import numpy as np
from ML_Project_Utilities import read_images, split_data, calc_fscore, write_model_report
from sklearn import svm
""" from sklearn.model_selection import GridSearchCV """
from sklearn.metrics import accuracy_score, recall_score, precision_score
import joblib    

# Reading images & labels
images, labels = read_images()

# Flatting images
images = np.array([img.flatten() for img in images])

# reformat labels
labels = np.argmax(labels, axis=1)


# Spliting data into training & testing
training_images, testing_images, training_labels, testing_labels = split_data(images, labels, 0.8)

"""
# Defining parameters
param_grid = [{
  'C': [0.1, 1, 10, 100],
  'gamma': [0.0001, 0.001, 0.1, 1],
  'kernel':['rbf','poly']
}]

# sv classifier
svc = svm.SVC()

# model
model = GridSearchCV(svc, param_grid)


# training model
model.fit(training_images, training_labels)

# getting the best estimator
model = model.best_estimator_

"""

# model classifier
model = svm.SVC(C=10, gamma=0.001)

# training model
print('Model fitting has been started.')

model.fit(training_images, training_labels)

print('Model fitting has been finished.')

# saving model 
joblib.dump(model, "model4.pkl")

# making prediction
predicted_labels = model.predict(testing_images)

# reporting model
with open("model4.txt", "w") as file: 
      file.write(f"{model}\n\n")
      
recall = recall_score(testing_labels, predicted_labels, average='micro')
precision = precision_score(testing_labels, predicted_labels, average='micro')

write_model_report("model4", {
  "accuracy": accuracy_score(testing_labels, predicted_labels),
  "recall": recall,
  "precision": precision,
  "fscore": calc_fscore(recall, precision)
})