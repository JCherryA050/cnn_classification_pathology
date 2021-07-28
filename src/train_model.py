# Standard Imports
import numpy as np

# Importing all relevant packages for modeling in keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Setting the random seed for reproducability
np.random.seed(123)
def initialize_generators(file_path,target_size):
    '''
    Function for initializing the data generators for test,train, and validation. 
    Takes a target image size as arguement.

    Arguments:

    target_size: tuple-type

    '''
    # Define paths to the data directories
    train_folder = file_path + 'train/'
    test_folder = file_path + 'test/'
    val_folder = file_path + 'validation/'

    try:
        # set up batch generator for train set
        train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
                train_folder, 
                target_size=target_size, 
                batch_size = 516)
        # set up batch generator for test set
        test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
                test_folder, 
                target_size=target_size, 
                batch_size = 32) 
        # set up batch generator for validation set
        val_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
                val_folder, 
                target_size=target_size, 
                batch_size = 64)
        return 1
    except:
        print('Files not found.')
        return 0
    
def run_model(backbone = None,model_path=None,model_name=None,target_size=None):

    '''
    Function for initializing and running keras models.

    Arguments:

    backbone: .h5, if None the model is set to the DenseNet201 pretrained model.
    model_path: str, path to save the trained model to. if None the model is saved in the same folder as the script calling the function
    model_name: str, name to save the model under. if None, default is the same directory as the master
    target_size: tuple - int, tuple with the target image size for rescaling. in None target size is set to (250,250)

    
    '''

    # Initialize default variables
    if not backbone:
        #define the default model
        backbone = DenseNet201(
            weights='imagenet',
            include_top=False,
            input_shape=target_size)
    else:
        pass    
    if model_path:
        pass
    else:
        model_path = '.'
    if model_name:
        pass
    else:
        model_name = 'model'
    if target_size:
        pass
    else:
        target_size = (250,250)

    # Creating admin tools for the models like automatic saving checkpoints, early stopping routines, etc.
    checkpoint = ModelCheckpoint(model_path+model_name+'.h5',
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                save_weights_only=False,
                                mode='auto',
                                period=1)

    early = EarlyStopping(monitor='val_acc',
                        min_delta=0,
                        patience=4,
                        verbose=1,
                        mode='auto')

    # Function used to build the model using the model argument
    def build_model(backbone, lr=1e-3):
        model = Sequential()
        model.add(backbone)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dropout(0.5))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(2, activation='softmax'))
        
        opt = Adam(lr=lr)# beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False
        model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['acc'])
        print(model.summary())
        return model

    # Set the model backbone.
    model = build_model(backbone)

    # Print the full model summary.
    print(model.summary())

    # Fit the model to the train data
    history = model.fit(train_generator,
                        epochs = 20,
                        steps_per_epoch=10,
                        validation_data=val_generator,
                        callbacks=[checkpoint],
                        class_weight=weights)#
    return model

