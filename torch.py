# Import Required libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import urllib.request
import tarfile

filename = "flower_photos.tgz"
tf = tarfile.open(filename)
tf.extractall()

# This is the directory path where all the class folders are
dir_path =  'flower_photos'
 
# Initialize classes list, this list will contain the names of our classes.
classes = []
 
# Iterate over the names of each class
for class_name in os.listdir(dir_path):
 
   # Get the full path of each class
  class_path = os.path.join(dir_path, class_name)
  
  # Check if the class is a directory/folder
  if os.path.isdir(class_path):
 
      # Get the number of images in each class and print them
      No_of_images = len(os.listdir(class_path))
      print("Found {} images of {}".format(No_of_images , class_name))
 
      # Also store the name of each class
      classes.append(class_name)
 
# Sort the list in alphabatical order and print it    
classes.sort()
print(classes)