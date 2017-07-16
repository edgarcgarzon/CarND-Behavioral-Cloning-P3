import csv
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from keras.layers import Input, Flatten, Dense, Cropping2D, Lambda, Conv2D
from keras.models import Model
import matplotlib.pyplot as plt
import time

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

def MultCamAugm(X, y):

    correction = 0.2

    Xa = []
    ya = []

    #Generate a flat list from left, Center and Right images
    for images, angle in zip(X,y):

        Xa.extend([images[0], images[1], images[2]])
        ya.extend([ float(angle) + correction, float(angle), float(angle) - correction])

    return Xa, ya

def FlipAugm(X, y):

    Xa = []
    ya = []

    #Generate a flat list from left, Center and Right images
    for images, angle in zip(X,y):

        Xa.extend([images[0], images[1], images[2]])
        ya.extend([ float(angle) + correction, float(angle), float(angle) - correction])

    return Xa, ya


def getImage(fileName):
    # Read image
    currentPath = os.path.join('./data/IMG/', fileName.split('/')[-1])
    imgBGR = cv2.imread(currentPath)
    #imgBGR = cv2.resize(imgBGR, (0, 0), fx=0.5, fy=0.5)
    return cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)


def flipImage(image):
    return cv2.flip(image, 1)


def generator(X, y, augm = False, batch_size=32):

    if augm == True:
        Xa, ya = MultCamAugm(X, y)
    else:
        Xa = [i[1] for i in X]
        ya = y

    Xa, ya = shuffle(Xa, ya)

    num_samples = len(Xa)
    print("Number of samples = ", num_samples)

    while 1:  # Loop forever so the generator never terminates

        for offset in range(0, num_samples, batch_size):

            images_batch = Xa[offset:offset + batch_size]
            angle_batch = ya[offset:offset + batch_size]

            images = []
            angles = []

            for im, angle in zip(images_batch, angle_batch):
                #print("Image = ", im, "   Angle = ", angle)
                #Load original images
                image = getImage(im)
                images.append(image)
                angles.append(float(angle))

            # Transform to np.arrays
            X_batch = np.array(images)
            y_batch = np.array(angles)


            yield shuffle(X_batch, y_batch)

def modelDefintion():

    input_shape = (160, 320, 3)
    inp = Input(shape=input_shape)

    x = Lambda(lambda x: x / 127.5 - 1.)(inp)
    x = Cropping2D(cropping=((50,20), (0,0)))(x)
    x = Conv2D(24, 5, 5, activation='relu', subsample=(2,2))(x)
    x = Conv2D(36, 5, 5, activation='relu', subsample=(2,2))(x)
    x = Conv2D(48, 5, 5, activation='relu', subsample=(2,2))(x)
    x = Conv2D(64, 3, 3, activation='relu')(x)
    x = Conv2D(64, 3, 3, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(100)(x)
    x = Dense(50)(x)
    x = Dense(10)(x)
    x = Dense(1)(x)

    m = Model(inp, x)
    m.summary()

    return m

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

    model = modelDefintion()

    model.compile(loss = 'mse', optimizer='adam')

    train_generator = generator(X_train, y_train, augm = True, batch_size = BATCH_SIZE)
    validation_generator = generator(X_val, y_val, augm = True, batch_size = BATCH_SIZE)

    t0 = time.time()

    stepsPerEpoch = 3*len(X_train)/BATCH_SIZE
    validationSteps = len(X_val)/BATCH_SIZE
    EPOCHS = 3

    historyObj = model.fit_generator(train_generator, steps_per_epoch = stepsPerEpoch, epochs = EPOCHS, verbose = 2,
                                  validation_data = validation_generator, validation_steps = validationSteps)

    print("Time: %.3f seconds" % (time.time() - t0))

    model.save('model.h5')

    plotLoss(historyObj)


if __name__ == '__main__':
    main()
