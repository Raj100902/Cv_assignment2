from flask import Flask, request, render_template, send_file
import os
import cv2
from werkzeug.utils import secure_filename
from algorithms import thresholding, kmeans_clustering, watershed, grabcut

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment():
    if 'file' not in request.files:
        return "No file uploaded!", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected!", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)

    image = cv2.imread(filepath)
    algo = request.form['algorithm']
    
    if algo == 'thresholding':
        binary, adaptive = thresholding(image)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'thresholding.png')
        cv2.imwrite(output_path, binary)
    elif algo == 'kmeans':
        segmented = kmeans_clustering(image)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'kmeans.png')
        cv2.imwrite(output_path, segmented)
    elif algo == 'watershed':
        segmented = watershed(image)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'watershed.png')
        cv2.imwrite(output_path, segmented)
    elif algo == 'grabcut':
        segmented = grabcut(image)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'grabcut.png')
        cv2.imwrite(output_path, segmented)
    else:
        return "Invalid algorithm selected!", 400

    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
