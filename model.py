import csv
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import matplotlib.pyplot as plt

BATCH_SIZE = 32

def LoadData(testSize):
    # https://docs.python.org/3/library/csv.html#csv.reader
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    currentPath = os.path.join('./data', 'driving_log.csv')

    X_train = []
    y_train = []

    with open(currentPath, newline='') as csvFile:
        csvReader = csv.DictReader(csvFile)

        for row in csvReader:
            X_train.append([row['left'], row['center'], row['right']])
            y_train.append(row['steering'])

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = testSize)

        return X_train, X_val, y_train, y_val

def generator(X, y, batch_size=32):

    num_samples = len(X)

    Xs, ys = shuffle(X, y)

    while 1:  # Loop forever so the generator never terminates

        for offset in range(0, num_samples, batch_size):

            images_batch = Xs[offset:offset + batch_size]
            angle_batch = ys[offset:offset + batch_size]

            images = []
            angles = []

            for im, angle in zip(images_batch, angle_batch):

                #Read image
                currentPath = os.path.join('./data/IMG/', im[1].split('/')[-1])
                center_image = cv2.imread(currentPath)
                images.append(center_image)

                #Read angle
                center_angle = float(angle)
                angles.append(center_angle)

            # Transform to np.arrays
            X_batch = np.array(images)
            y_batch = np.array(angles)

            #TODO: perform any transformation here before output

            yield X_batch, y_batch

def modelDefintion():

    input_shape = (160, 320, 3)
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    x = Dense(1)(x)

    return Model(inp, x)

def plotLoss(obj):
    ### print the keys contained in the history object
    print(obj.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(obj.history['loss'])
    plt.plot(obj.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


def main():

    #Load the file names for the train and validation features
    X_train, X_val, y_train, y_val = LoadData(testSize = 0.2)

    print("Number of samples = ", len(X_train))

    model = modelDefintion()

    model.compile(loss = 'mse', optimizer='adam')

    train_generator = generator(X_train, y_train, batch_size = BATCH_SIZE)
    validation_generator = generator(X_val, y_val, batch_size = BATCH_SIZE)

    historyObj = model.fit_generator(train_generator, steps_per_epoch = len(X_train)/BATCH_SIZE, epochs = 3, verbose = 2,
                                     validation_data = validation_generator, validation_steps = len(X_val)/BATCH_SIZE)

    model.save('model.h5')

    plotLoss(historyObj)



if __name__ == '__main__':
    main()
