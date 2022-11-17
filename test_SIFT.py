import os
import numpy as np
from flask import Flask, render_template, send_file

count = 0
app = Flask(__name__, template_folder = 'Template')
database_dir = "image.orig"


@app.route('/')
def index():
	return 'happy world!'
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'

if __name__ == '__main__':
      app.run(host='127.0.0.1', port=8000)


# Compute pixel-by-pixel difference and return the sum
