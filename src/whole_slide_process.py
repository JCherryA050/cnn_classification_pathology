# Importing Pillow to handle the concatenation
from PIL import Image, ImageColor

# Importing glob to import the images
from glob import glob

# Importing cv to convert the image paths into images
import cv2

# Importing os to separate and process the image file names
import os

# Standard Imports
import numpy as np

# Importing all relevant packages for modeling in keras
from tensorflow.keras.preprocessing import image



def concat(key_word, slide_number, model):
    # Loading in the images of one scan 8863
    data = glob('./data/IDC_regular_ps50_idx5/'+slide_number+'/**/*.png', recursive=True)
    
    # Separating file name from the path of the file
    files = []
    for datum in data:
            files.append(os.path.basename(datum))
    
    # removing the .png from the file names and isolating the x, y positions of the images
    x = []
    y = []
    for file in files:

        # isolating the x and y coordinates of the image and converting to int type
        x.append(int(file.split('_')[2].replace('x','')))
        y.append(int(file.split('_')[3].replace('y','')))
        
        
    # Initialize the full image space
    full_slide = Image.new('RGB',(max(x)-min(x),max(y)-min(y)),color='#f2f2f5')
    
    for datum in data:
        # Isolate the file name
        file = os.path.basename(datum)
        
        # grab the location of the image from the file
        x = int(file.split('_')[2].replace('x',''))
        y = int(file.split('_')[3].replace('y',''))
        case = file.split('_')[-1].replace('.png','')
        
        # Load the image in using the cv library
        img = Image.open(datum)
        
        # paste the image into the image space
        full_slide.paste(img,(x-51,y-51))
        
        if key_word == 'class':
            # paste the image into the image space
            if case == 'class0': 
                img_neg = Image.new('RGB',(50,50),color="#77c128")
                full_slide.paste(img_neg,(x-51,y-51))
            
            else:
                img_pos = Image.new('RGB',(50,50),color='#dd5866')
                full_slide.paste(img_pos,(x-51,y-51))
        elif key_word == 'image':
            img = Image.open(datum)
            full_slide.paste(img,(x-51,y-51))
        elif key_word == 'predict':
            # Function loading in an image from the dataset and making a prediction
            def make_prediction(img_path,target_size):
                img = image.load_img(img_path,target_size=target_size)
                img_array = image.img_to_array(img)
                img_batch = np.expand_dims(img_array,axis=0)
                # img_preprocessed = preprocess_input(img_array)
                return float(model.predict(img_batch/255))

            prediction = make_prediction(datum,(100,100))
            img = Image.new('RGB',(50,50),color=(int(255*prediction),int(255*prediction),int(255*prediction)))
            full_slide.paste(img,(x-51,y-51))

        else:
            print('not a valid key word')
            break
            
    return full_slide