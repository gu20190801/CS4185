from flask import Flask, render_template, send_file

count = 0
app = Flask(__name__, template_folder = 'templates')
database_dir = "image.orig"


@app.route('/')
def index():
	return 'happy world!'

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8000)

