import cv2
import numpy as np

class EigenFacesRecognition():
    def __init__(self, images, labels, num_eigen_faces = 10, threshold = 1e3):
        self.images = images
        self.num_eigen_faces = num_eigen_faces
        self.threshold = threshold * num_eigen_faces
        self.labels = labels
        self.means = []
        self.eigens = []
        self.__build()

    def predict1(self, image):
        min_score = self.threshold
        label = ''
        image = np.array(image).reshape(-1, 1)
        print(np.array(self.means).shape)
        for i in range(len(np.unique(self.labels))):
            omega = np.dot(self.eigens, image)
            thresh = np.sum(np.abs(omega - self.means[i]))
            if thresh < min_score:
                min_score = thresh
                label = np.unique(self.labels)[i]
        return label


    def predict(self, images):
        images = np.array(images).reshape(len(images), -1).T
        omega = np.dot(self.eigens, images)
        omega = omega.reshape((1, omega.shape[0], omega.shape[1]))
        thresh = np.abs(np.subtract(omega, self.means))
        sums = np.sum(thresh, axis = 1)
        args = np.argmin(sums, axis = 0)
        preds = np.unique(self.labels)[args]
        return preds

    def __create_data_matrix(self):
        '''
        Allocate space for all images in one data matrix.
            The size of the data matrix is
            ( w  * h  * 3, numImages )

            where,

            w = width of an image in the dataset.
            h = height of an image in the dataset.
            3 is for the 3 color channels.
            '''

        numImages = len(self.images)
        self.images = np.array(self.images).reshape(numImages, -1)


    def __build(self):
        # Size of images
        sz = self.images[0].shape
        # Create data matrix for PCA.
        self.__create_data_matrix()

        _, eigenVectors = cv2.PCACompute(self.images, mean=None, maxComponents=self.num_eigen_faces)
        for eigenVector in eigenVectors:
            eigenFace = eigenVector
            self.eigens.append(eigenFace)

        for label in np.unique(self.labels):
            # Compute the eigenvectors from the stack of images created
            class_data = self.images[np.argwhere(self.labels == label)]
            class_data = class_data[:, 0, :]

            mean, _ = cv2.PCACompute(class_data, mean=None, maxComponents=self.num_eigen_faces)

            self.means.append(np.dot(self.eigens, mean.T))

        self.images = []

        print("Model was sucessfully build")
