'''
Created on Mar 30, 2017

@author: Originally Joe Bockhorst, modified by Jude Shavlik
'''
import os, random
import numpy as np

# Want repeatable, yet (pseudo) random results.  Seems people are having trouble getting repeatable results (as am I).
np.random.seed(638 * 838) # See https://groups.google.com/forum/#!topic/keras-users/TObgSMN3bNY (recommended by the Keras developer)
random.seed(   638 + 838) # I also tried this, but since get different results each run.  That's too bad, since replication is very valuable!

from os.path                    import join, basename
from skimage.transform          import resize
from matplotlib.pyplot          import imread
from datetime                   import datetime
#import tensorflow as tf

from tensorflow.keras.models               import Sequential
from tensorflow.keras.layers               import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers               import MaxPooling2D
from tensorflow.keras.layers               import Conv2D, ZeroPadding2D
from tensorflow.keras.optimizers           import Adam
from tensorflow.keras.layers               import LeakyReLU

# TODO: extend to ensemble (where the difference is solely due to the seeds, 
#                           but could vary HUs, plates, and other parameters)

# TODO: save models to disk and recover when restarting(in some commented-out code I experimented with this unsuccessfully)

# TODO: run in Condor (eg, read a condorID, use it to determine parameters,
#                      save to disk, checkpoint periodically)

# TODO: allow 'flag' that says how many color channels to use

# TODO: add code for rotating, flipping, shifting images (ie, create extra training examples).

# TODO: write a 'one-layer of HUs' version?  A perceptron version?

# TODO: put helper functions in a separate file (I kept in one for simplicity in this posting)

# Can one use dropOut on the INPUT units? See https://github.com/fchollet/keras/issues/96


############################################################
###########     Supporting functions.    ###################  
############################################################    

def accuracy(model, X,  y):
    "Return the accuracy of model on Inputs X and labels y."
    predict_x =model.predict(X) 
    y_hat = np.argmax(predict_x, axis=1)
    #y_hat = model.predict_classes(X, verbose=0)
    n_correct = (np.array(y_hat) == np.array(y)).sum()
    return n_correct / float(y.size)

def prod(z):
    "Return the product z[0] * z[1] * ... * z[-1]."
    # Jude get a complaint on this so rewrote: return reduce(lambda x, y: x*y, z, 1)
    result = 1
    for i in range(0, len(z)):
        result *= z[i]
    return result

def load_course_images(directory, image_size):
    """
    load the cs638 course images and labels from a directory.
    image_size = (L, W) the length and width of returned images
    
    return X, y, y_onehot
    
        X - a numpy ndarray with shape (n, L, W, 4). 
            n is the # of images. 
            L,W is imageIndex length and width
            4 is for the RGB + Grayscale channels
            
        y - 1D numpy array of imageIndex labels encoded as integer between 0 and 5
        
        y_onehot - one-hot encoding of y as numpy array with shape (n, 6)
    """
    labels = []
    files  = [f for f in os.listdir(directory) if f.endswith("jpg")]
    shape  = [len(files)] + [image_size, image_size, 4]
    X      = np.zeros(shape = shape) + np.nan  # Put a NaN in each cell to make sure we replace it below?
    
    for imageIndex, f in enumerate(files):
        I = imread("{}/{}".format(directory, f))
        I = resize(I, [image_size, image_size])
        X[imageIndex, :, :, 0:3] = I / 255.0                # rgb  channels
        X[imageIndex, :, :, 3]   = (I.mean(axis=2) / 255.0) # gray channel
        labels.append(f.split("_")[0]) # Pull the LABEL out of the image name (i.e., all the characters before the first underscore).
        
    assert np.isnan(X).sum() == 0 # Make sure no NaNs remain in the data.
    y        = np.array([LABELS.index(lbl) for lbl in labels]) # Collect all the LABELs.
    y_onehot = np.zeros((X.shape[0], len(LABELS))) # Now need to convert to a 'one hot' representation.
    for rowIndex, colIndex in enumerate(y):  # This will produce ( (0, y(0)), (1, y(1)), ..., (n, y(n)) ) where n+1 is the number of images.
        y_onehot[rowIndex, colIndex] = 1 # For each image, set one output unit to 1.
        
    return X, y, y_onehot, [join(directory, f) for f in files]


############################################################

# BE SURE TO EDIT THE LINE BELOW TO THE DIRECTORY WHERE YOU ARE STORING THE LAB3 IMAGES!
IMG_DIR           = "../data/images"

LABELS            = ['airplanes', 'butterfly', 'flower', 'grand', 'starfish', 'watch']   
imageDimension    = 32
numberOfColors    =   4 # R, G, B, and Gray
epochsToRun       = 100
batch_size        =  10 # how many gradients we collect before updating weights
platesConv1       =  16
platesConv2       =  16
kernelSizeConv1   =   4
kernelSizePool1   =   2
kernelSizeConv2   =   4
kernelSizePool2   =   2
strideConv1       =   1 # same for both x and y dimensions
stridePool1       =   2 # same for both x and y dimensions
strideConv2       =   1 # same for both x and y dimensions
stridePool2       =   2 # same for both x and y dimensions
zeroPaddingConv1  =   1 # same for both x and y dimensions (this is padding of the INPUT image)
zeroPaddingPool1  =   1 # same for both x and y dimensions (this is padding of the CONV1 image)
zeroPaddingConv2  =   1 # same for both x and y dimensions (this is padding of the POOL1 image)
zeroPaddingPool2  =   1 # same for both x and y dimensions (this is padding of the CONV2 image)
input_dropoutProb =   0.05
conv1_dropoutProb =   0.50
pool1_dropoutProb =   0.00
conv2_dropoutProb =   0.50
pool2_dropoutProb =   0.00
final_dropoutProb =   0.50
numberOfFinalHUs  = 128
numberOfClasses   = len(LABELS)

# Early stopping.
confusionTestsetAtBestTuneset = np.zeros((numberOfClasses, numberOfClasses))
bestTuneSetEpoch     = np.nan
bestTuneSetAcc       = 0
testSetAccAtBestTune = 0

# Read in the images.  Only the TEST examples really need to be kept (at the end a web page of testset errors produced).
X_train, y_train, y_onehot_train, img_files_train = load_course_images(directory="{}/trainset".format(IMG_DIR), image_size=imageDimension)
X_tune,  y_tune,  y_onehot_tune,  img_files_tune  = load_course_images(directory="{}/tuneset".format( IMG_DIR), image_size=imageDimension)
X_test,  y_test,  y_onehot_test,  img_files_test  = load_course_images(directory="{}/testset".format( IMG_DIR), image_size=imageDimension)
print("There are {:,} training examples.".format(len(X_train)))
print("There are {:,} tuning examples.".format(  len(X_tune)))
print("There are {:,} testing examples.".format( len(X_test)))


# Define model architecture  See https://keras.io/getting-started/sequential-model-guide/
model = Sequential()
#model.add(Dropout(input_dropoutProb)) # Can't specify dropOut for input units?  See https://github.com/fchollet/keras/issues/96

leakyReLUtoUse = LeakyReLU(alpha = 0.1)
model.add(Conv2D(platesConv1, 
                 kernel_size = kernelSizeConv1,
                 input_shape = [imageDimension, imageDimension, numberOfColors],
                 data_format = "channels_last", # Says that the color channels are LAST.
                 strides     = strideConv1, 
                 padding     = "valid", # I'm not sure what this does?  Says zero padding is ok????
                 use_bias    = True))
model.add(leakyReLUtoUse); # Have to add as a layer, not as an argument to Conv2D.  See https://github.com/fchollet/keras/issues/3380
model.add(ZeroPadding2D(padding = zeroPaddingConv1, data_format = "channels_last"))
model.add(Dropout(conv1_dropoutProb)) 

model.add(MaxPooling2D(pool_size = kernelSizePool1, strides = stridePool1, padding = 'valid'))
model.add(Dropout(pool1_dropoutProb))
model.add(ZeroPadding2D(padding  = zeroPaddingPool1))

model.add(Conv2D(platesConv2, 
                 kernel_size = kernelSizeConv2,
                 strides     = strideConv2, 
                 padding     = "valid", # zero padding????
                 use_bias    = True))
model.add(leakyReLUtoUse); # Have to add as a layer, not as an argument to Conv2D.  See https://github.com/fchollet/keras/issues/3380
model.add(Dropout(conv2_dropoutProb))
model.add(ZeroPadding2D(padding=  zeroPaddingConv2))

model.add(MaxPooling2D(pool_size = kernelSizePool2, strides = stridePool2, padding = 'valid'))
model.add(Dropout(pool2_dropoutProb))
model.add(ZeroPadding2D(padding  = zeroPaddingPool2))

model.add(Flatten()) # Flattens the last MAX POOL layer so can fully connect to the final HU layer.

model.add(Dense(units = numberOfFinalHUs))
model.add(Activation('relu'))
model.add(Dropout(final_dropoutProb))

model.add(Dense(units = numberOfClasses))
model.add(Activation("softmax"))

# Report the size of the model.
# numberOfWeights = 0
# for tw in model.trainable_weights:
#     numberOfWeights += prod([dim.value for dim in tw.get_shape()])
# print()
# print("The model has {:,} trainable weights.".format(numberOfWeights))
# print()

### Train the model.

optimizerToUse = Adam() # Use the defaults (https://keras.io/optimizers/).
model.compile(loss = 'categorical_crossentropy', optimizer = optimizerToUse, metrics = ['accuracy']) # Had been 'rmsprop' (good for recurrent nets).
for i in range(epochsToRun):
    model.fit(X_train, 
              y_onehot_train, 
              epochs     = 1, 
              batch_size = batch_size,
              verbose    = 0,  # 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoc
              #validation_data=(X_tu, y_onehot_tu),  
              shuffle    = True)
              
    acc_train = accuracy(model, X_train, y_train)
    acc_tune  = accuracy(model, X_tune, y_tune)
    acc_test  = accuracy(model, X_test, y_test)
    
    marker    = " "
    if acc_tune > bestTuneSetAcc:
        bestTuneSetEpoch     = i
        bestTuneSetAcc       = acc_tune
        testSetAccAtBestTune = acc_test
        marker               = "*"                
        #y_pred_test = model.predict_classes(X_test, verbose = 0)
        predict_x =model.predict(X_test) 
        y_pred_test = np.argmax(predict_x, axis=1)

        confusionTestsetAtBestTuneset.fill(0)
        for y_pred, y in zip(y_pred_test, y_test): # EXAMPLE: zip((a, b, c), (x, y, z)) produces ( (a, x), (b, y), (c, z) ).
            confusionTestsetAtBestTuneset[y, y_pred] += 1
#         # The following had problems discussed here (I updated h5py, but that didn't fix the problem; even restarted LiCLipse): https://github.com/fchollet/keras/issues/6005
#         # Serialize model to JSON (from http://machinelearningmastery.com/save-load-keras-deep-learning-models/).
#         model_json = model.to_json()
#         with open("best_model.json", "w") as json_file:
#             json_file.write(model_json)
#         # Serialize weights to HDF5. NOTE: could just create and save the confusionTestsetAtBestTuneset matrix in RAM, but we need to save to disk for checkpointing anyway.
#         model.save_weights("best_model.h5", overwrite=True) # Would like to simply save in memory ...
#     
    print("After {:>4} epochs, accuracies are:  train = {:.3f}  tune = {:.3f}{} test = {:.3f}    {}".format(i+1, acc_train, acc_tune, marker, acc_test, datetime.now().strftime('%H:%M:%S  %m-%d-%Y')))
    
# Done with training.
print()
print("Final accuracy train = {:.3f}, tune = {:.3f}, test = {:.3f}".format(acc_train, acc_tune, acc_test))

# # Recover the best model.
# json_file = open('bestmodel.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# bestModel = model_from_json(loaded_model_json)
# bestModel.load_weights("best_model.h5")
# print("Loaded best model from disk.")
# bestModel.compile(loss = 'categorical_crossentropy', optimizer = optimizerToUse, metrics = ['accuracy'])
# 
# # Create a confusionTestsetAtBestTuneset matrix.
# confusionTestsetAtBestTuneset   = np.zeros((numberOfClasses, numberOfClasses))
# y_pred_test = bestModel.predict_classes(X_test, verbose = 0)
# for y_pred, y in zip(y_pred_test, y_test): # EXAMPLE: zip((a, b, c), (x, y, z)) produces ( (a, x), (b, y), (c, z) ).
#     confusionTestsetAtBestTuneset[y, y_pred] += 1
    
print()
print("Best testset accuracy at Epoch #{} = {:.3f}".format(bestTuneSetEpoch+1, testSetAccAtBestTune))
print()
print("Testset confusionTestsetAtBestTuneset matrix chosen by early stopping (rows are true class, cols are predicted class)")
print()
fmt = " | {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
print("{:10}".format("") + fmt.format(*LABELS))
print("------------------------------------------------------------------------------")
for rowIndex in range(numberOfClasses):
    print("{:10}".format(LABELS[rowIndex]) + fmt.format(*confusionTestsetAtBestTuneset[rowIndex, :]))

### Create a web page to view the errors.
with open(join(IMG_DIR, "errors.html"), "w") as F:
    F.write("<table> \n".format(LABELS[y_pred]))
    for y_pred, y, filename in zip(y_pred_test, y_test, img_files_test): # See zip example above.
        if y_pred != y:
            img_file = join("testset", basename(filename))
            F.write("<tr> <td>I thought this was a {}</td>\n".format(LABELS[y_pred]))
            F.write("<td> <img src='{}' style='border-color: red' border=5> </td>\n".format(img_file))
            F.write("</tr>\n")