# -*- coding: utf-8 -*-
“””
3D Medicare Image Classification
3D CNN to perform organ tissue segmentation from volumetric 3D medical images.

Functions:
main - main function, load data, create model and test performance of the models
getMatrix - get image matrix and labels
loadData - load data from files
createModel - create CNN models
plot - plot the results calculated from the models

Created on Sunday Apr 10:23:13 2019
Author: Hongya Lu
“””

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D

from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing



# image specification
img_rows, img_cols, img_depth = 16, 16, 15

# Training data
X_tr = []           # variable to store entire dataset
label = []          # label of movements

def main():
    X_tr, label = getMatrix()
    X_tr_array = np.array( X_tr )   # convert the frames read into array
    label = np.array( label )
    num_samples = len( X_tr_array )
    print( num_samples )
    train_data = [ X_tr_array, label ]
    ( X_train, y_train ) = ( train_data[ 0 ], train_data[ 1 ] )
    print( 'X_Train shape:', X_train.shape )
    train_set = np.zeros( ( num_samples, 1, img_rows, img_cols, img_depth ) )
    for h in range( num_samples ):
        train_set[ h ][ 0 ][ : ][ : ][ : ] = X_train[ h, :, :, : ]

    # Pre-processing
    train_set = train_set.astype( 'float32' )
    train_set -= np.mean( train_set )
    train_set /= np.max( train_set )
    Y_train = np_utils.to_categorical( y_train, nb_classes ) # convert class vectors to binary class matrices

    # Split the data
    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split( train_set, Y_train, test_size = 0.2, random_state = 4 )

    # Train the model
    model = createModel()
    hist = model.fit( X_train_new, y_train_new, validation_data = ( X_val_new, y_val_new ), batch_size = batch_size, nb_epoch = nb_epoch, shuffle = True )

    # Save model
    # model.save( ‘current.h5’ )

    # Evaluate the model
    score = model.evaluate( X_val_new, y_val_new, batch_size = batch_size, #show_accuracy=True )
    print( '**********************************************' )
    print( 'Test score:', score )
    print( 'History', hist.history )
    plot( hist )



# reading 6 types of movement, including boxing, hand clapping, handwaving, jogging, running and walking
def getMatrix():
    # Reading boxing action class
    items = [ ‘boxing’, ‘handclapping’, ‘handwaving’, ‘jogging’, ‘running’, ‘walking’ ]
    for i, item in enumerate( items ):
        pathname = ‘kth-dataset/‘ + item
        filenames = os.listdir( pathname )
        loadData( pathname, filenames, i )
        t = item + ‘ done’
        print( t )
    return X_tr, label

def loadData( pathname, filenames, c ):
    for vid in filenames:
        vid = pathname + vid
        frames = []
        cap = cv2.VideoCapture( vid )
        fps = cap.get( 5 )
        # print( ‘Frames per second using video.get( cv2.cv.CV_CAP_PROP_FPS ): {0}’.format( fps ) )

        for k in xrange(15):
            ret, frame = cap.read()
            frame = cv2.resize( frame, ( img_rows, img_cols ), interpolation = cv2.INTER_AREA )
            gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
            frames.append( gray )
            if cv2.waitKey(1) & 0xFF == ord( 'q' ):
                break
        cap.release()
        cv2.destroyAllWindows()
        input = np.array( frames )
        ipt = np.rollaxis( np.rollaxis( input, 2, 0 ), 2, 0 )
        X_tr.append( ipt )
	label.append( c )

def createModel():
    
    # model structure
    # number of convolutional filters, level of pooling ( POOL x POOL ) and level of convolution ( CONV x CONV ) to perform at each layer
    nb_filters = [ 32, 32 ]
    nb_pool = [ 3, 3 ]
    nb_conv = [ 5, 5 ]

    # CNN Training parameters
    batch_size = 2
    nb_classes = 6
    nb_epoch = 50
    patch_size = 15  # img_depth or number of frames used for each video

    # Define model
    model_exists = os.path.exists( 'current.h5' )
    if ( model_exists ):
        model = load_model( 'current.h5' )
        print( ‘**************************************************’ )
        print( ‘current.h5 model loaded’)
    else:
        model = Sequential()
        model.add( Convolution3D( nb_filters[0], 
        kernel_dim1 = nb_conv[0], # depth
        kernel_dim2 = nb_conv[0], # rows
        kernel_dim3 = nb_conv[0], # cols
        input_shape = ( 1, img_rows, img_cols, patch_size ),
        activation = 'relu' 
	) )
        model.add( MaxPooling3D( pool_size = ( nb_pool[0], nb_pool[0], nb_pool[0] ) ) )
        model.add( Dropout( 0.5 ) )
        model.add( Flatten() )
        model.add( Dense( 128, init = 'normal', activation = 'relu' ) )
        model.add( Dropout( 0.5 ) )
        model.add( Dense( nb_classes, init = 'normal' ) )
        model.add( Activation( 'softmax' ) )
        model.compile( loss = 'categorical_crossentropy', optimizer = 'RMSprop', metrics = [ 'mse', 'accuracy' ] )
    return model

# Plot the results
def plot( hist ):
    train_loss = hist.history[ 'loss' ]
    val_loss = hist.history[ 'val_loss' ]
    train_acc = hist.history[ 'acc' ]
    val_acc = hist.history[ 'val_acc' ]
    xc = list( range( 100 ) )

    plt.figure( 1, figsize = ( 7,5 ) )
    plt.plot( xc, train_loss ) 
    plt.plot( xc, val_loss )
    plt.xlabel( 'num of Epochs' )
    plt.ylabel( 'loss' )
    plt.title( 'train_loss vs val_loss' )
    plt.grid( True )
    plt.legend( [ 'train', 'val' ] )
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use( [ 'classic' ] )

    plt.figure( 2, figsize = ( 7,5 ) )
    plt.plot( xc, train_acc )
    plt.plot( xc, val_acc )
    plt.xlabel( 'num of Epochs' )
    plt.ylabel( 'accuracy' )
    plt.title( 'train_acc vs val_acc' )
    plt.grid( True )
    plt.legend( [ 'train', 'val' ], loc = 4 )
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use( [ 'classic' ] )
