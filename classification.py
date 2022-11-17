import numpy as np
import pandas as pd 
import os
from sklearn.metrics import classification_report
import seaborn as sn; sn.set(font_scale = 1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf
from tqdm import tqdm

class_names = ['beach', 'building', 'bus', 'dinosaur', 'flower', 'horse', 'man']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

print(class_names_label)

img_size = (150, 150)

def load_data():
	directory = r"C:\Users\s1410\Downloads\2022-Codes-Python"
	category = ["image.orig"]

	output = []

	for c in category:
		path = os.path.join(directory, c)
		print(path)
		images = []
		labels = []

		print("loading {}".format(c))

		for folder in os.listdir(directory):
			label = class_names_label[folder]

			for file in os.listdir(os.path.join(os.path, folder), file):
				img_path = os.path.join(os.path.join(path, folder), file)
				img = cv.imread(img_path)
				img = cv.cvtColor(img, cv.COLOR_RGB2RGB)
				img = cv.resize(img, img_size)

				images.append(img)
				labels.append(label)

			images = np.array(images, dtype='float32')
			labels = np.array(labels, dtype='int32')

			output.append((images, labels))

		return output

	(train_images, train_labels), (test_images, test_labels) = load_data()

	train_images, train_labels = shuffle(train_images, train_labels, random_state = 25)

	def display_example(class_names, images, labels):
		figsize = 20, 20
		fig = plt.figure(figsize=figsize)
		for i in range(25):
			plt.subplot(5, 5, i+1)
			plt.xticks([])
			plt.yticks([])
			plt.grid(False)
			image = cv.resize(images[i], figsize)
			plt.imshow(image.astype(np.uint8))
			plt.xlabel(class_names[labels[i]])
		plt.show()
	display_examples(class_names, train, images, train_labels)

load_data()