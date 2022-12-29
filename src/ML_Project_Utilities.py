# Reading images & labels
import numpy as np
import cv2 as cv
import glob
from sklearn.model_selection import train_test_split
from tensorflow import keras



""" reading images function """
def read_images(gray_scale=False):
    print('Reading images...')
    images = []
    labels = []

    for label in range(0, 10):
        path = f"./Dataset/{label}/*.JPG"
        
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
            labels.append([1 if i == label else 0 for i in range(0, 10)])

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



""" calculating fscore """
def calc_fscore(recall, precision):
    return 2 * ((recall * precision) / (recall + precision))



""" splitting data """
def split_data(features, output, train_size): 
    return train_test_split(features, output, train_size=train_size)



""" writing model report """
def write_model_report(file_name, measurements={}):
    with open(f"{file_name}.txt", "a") as file: 
        for m in measurements:    
            file.write(f"{m} = {measurements[m]}\n")



""" Running model """
def run_nn_model(model_id, model: keras.models.Sequential, images, labels, train_size=0.8, epochs=30):
    # Showing model summary
    with open(f"{model_id}.txt", "w") as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))
        file.write('\n')
    
    # Spliting data into training & testing
    training_images, testing_images, training_labels, testing_labels = split_data(images, labels, train_size)

    # Configuring model
    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy', keras.metrics.Recall(), keras.metrics.Precision()]
    )
    
    # Fitting model
    model.fit(
        training_images, 
        training_labels, 
        epochs=epochs, 
        verbose=1,
        validation_data=(testing_images, testing_labels)
    )
    
    # Evaluating model
    loss, accuracy, recall, precision = model.evaluate(testing_images, testing_labels, verbose=2)
    
    # Saving model
    model.save(f"{model_id}.h5")
    
    # Reporting model 
    write_model_report(model_id, {
        "loss": loss,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "fscore": calc_fscore(recall, precision)
    })