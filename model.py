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
import argparse

BATCH_SIZE = 128
data_dir = ''
EPOCHS = 3


def LoadData(testSize = 0.2):
    # https://docs.python.org/3/library/csv.html#csv.reader
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    print("Loading data from: ", data_dir)
    currentPath = os.path.join(data_dir, 'driving_log.csv')

    X_train = []
    y_train = []

    with open(currentPath, newline='') as csvFile:
        csvReader = csv.reader(csvFile)

        for row in csvReader:
            #skip header
            if(row[0] == 'center'):
                next(csvReader)
            else:
                X_train.append([row[0], row[1], row[2]])
                y_train.append(row[3])


        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = testSize)

        return X_train, X_val, y_train, y_val

def MultCamAugm(X, y):
    '''
    Data augmentation function for using the images of the left and center cameras
    :param X: features: [Left image, center image, right images]
    :param y: Labels: angle
    :return: X, y: data augmented
    '''

    correction = 0.2

    Xa = []
    ya = []

    #Generate a flat list from left, Center and Right images
    for images, angle in zip(X,y):

        Xa.extend([images[0], images[1], images[2]])
        ya.extend([ float(angle), float(angle) + correction, float(angle) - correction])

    return Xa, ya

def getImage(fileName):
    # Read image
    currentPath = os.path.join(data_dir, fileName.strip())
    imgBGR = cv2.imread(currentPath)
    #imgBGR = cv2.resize(imgBGR, (0, 0), fx=0.5, fy=0.5)
    return cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)


def generator(X, y, augm = False, batch_size=32):
    '''

    :param X: features
    :param y: Labels
    :param augm: data augmentation enable
    :param batch_size: batch size
    :return: batch
    '''

    if augm == True:
        Xa, ya = MultCamAugm(X, y)
    else:
        Xa = [i[0] for i in X]
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
    '''
    Nvidia model definition
    :return: model
    '''

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
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(1)(x)

    m = Model(inp, x)
    m.summary()

    return m

def modelFineTuning(m, modelLoadFile):
    '''
    https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html
    :param m: model
    :param modelLoadFile: model load file
    :return: model to train
    '''

    #redefine the inputs
    input_shape = (160, 320, 3)
    inp = Input(shape=input_shape)

    #load the last weights
    m.load_weights(modelLoadFile)

    #Remove the last layer
    m.layers.pop()
    m.layers[-1].outbound_nodes = []

    #Add new fully connected layer
    x = Dense(1)(m.layers[-1].output)

    #define a new model for fine tuning
    modelFT = Model(inputs=m.input, outputs=x)

    print("fine tunning ... !!!!!!")
    modelFT.summary()

    #return the new model to train
    return modelFT


def plotLoss(obj):
    '''
    :param obj: History object returned from fit or fit_generator function
    '''

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

def str2bool(v):
    return v.lower().strip() in ("yes", "true", "t", "1")

def main():

    parser = argparse.ArgumentParser(description='Behavioral Cloning P3')
    parser.add_argument('-d',    '--data_dir',        help='train data directory',       default='./data')
    parser.add_argument('-sf',   '--model_save_file', help='file to save the model',     default='model.h5')
    parser.add_argument('-ft',   '--fine_tuning',     help='fine tuning option',         default='False')
    parser.add_argument('-lf',   '--model_load_file', help='fine tuning data directory', default='model.h5')
    args = parser.parse_args()

    if True:
        print(args.data_dir)
        print(args.model_save_file)
        print(args.fine_tuning)
        print(args.model_load_file)

    global data_dir
    data_dir = args.data_dir

    #Load the file names for the train and validation features
    X_train, X_val, y_train, y_val = LoadData(testSize = 0.2)

    #Model definition
    model = modelDefintion()

    #Model fine tuning
    if(str2bool(args.fine_tuning) == True):
        model = modelFineTuning(model, args.model_load_file)

    #model compilation
    model.compile(loss = 'mse', optimizer='adam')

    #define the train an validation generators
    train_generator = generator(X_train, y_train, augm = True, batch_size = BATCH_SIZE)
    validation_generator = generator(X_val, y_val, augm = False, batch_size = BATCH_SIZE)

    t0 = time.time()

    stepsPerEpoch = 3*len(X_train)/BATCH_SIZE  # The data is augmented with the 3 cameras
    validationSteps = len(X_val)/BATCH_SIZE    # The validation data is keep to the original size

    #Train the model
    historyObj = model.fit_generator(train_generator, steps_per_epoch = stepsPerEpoch, epochs = EPOCHS, verbose = 1,
                                  validation_data = validation_generator, validation_steps = validationSteps)

    print("Time: %.3f seconds" % (time.time() - t0))

    #Save model
    model.save(args.model_save_file)

    #plot results
    plotLoss(historyObj)

if __name__ == '__main__':
    main()
