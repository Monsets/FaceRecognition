import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from Eigenfaces import EigenFacesRecognition

path_to_dataset = 'datasets/ATAT/att_faces/orl_faces'
path_to_test_images = 'datasets/ATAT/att_faces/test'

NUM_EIGEN_FACES = 10

def readImages():
    dirs = [d for d in os.listdir(path_to_dataset) if os.path.isdir(os.path.join(path_to_dataset, d))]

    images = []
    labels = []

    for dir in dirs:
        files = os.listdir(os.path.join(path_to_dataset, dir))
        for file in files:
            images.append(cv2.imread(os.path.join(os.path.join(path_to_dataset, dir), file)))
            labels.append(dir)

    return np.array(images), np.array(labels)

def load_test_images(path):
    files = os.listdir(path)

    images, labels = [], []

    for file in files:
        images.append(cv2.imread(os.path.join(path, file)))
        labels.append(os.path.splitext(file)[0])

    return images, labels


# Read images
images, labels = readImages()

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.10)
print("Длина тренировки: {} Длина теcта: {}".format(len(x_train), len(x_test)))

efr = EigenFacesRecognition(x_train, y_train, num_eigen_faces = 300)

preds = efr.predict(x_test)

print("Total error: {}".format(len(y_test) - np.sum(np.array(preds == y_test).astype(int))))