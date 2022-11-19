import cv2 as cv
import os
import numpy as np
from flask import Flask, render_template, send_file
from glob import glob

count = 0
app = Flask(__name__, template_folder = 'Template')
database_dir = "image.orig"


@app.route('/')
def index():
	image_name = 'horse.jpg'
	directory = image_name[0:len(image_name)-4]+"/"
	static = "static/"
	#parent_dir = "2022-Codes-Python/"
	#image_path = parent_dir+image_name
	#static_path = os.path.join(parent_dir, static)

	#isExist = os.path.exists(static_path)
	#if not isExist:
		#os.mkdir(static_path)

	#path = os.path.join(static_path, directory)
	#isExist = os.path.exists(path)
	#if not isExist:
		#os.mkdir(path)

	#img = cv.imread(image_name)
	
	global count
	#os.chdir(path)
	#cv.imwrite(str(count)+'.jpg', img)
	count+=1
			
	imageList = os.listdir(static+directory)
	imagelist = [directory + image for image in imageList]
	return render_template("index.html", imagelist=imagelist, count=count)
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8000)


# Compute pixel-by-pixel difference and return the sum
