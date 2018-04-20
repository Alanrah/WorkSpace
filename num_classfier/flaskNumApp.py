# -*- coding: utf-8 -*-
import os
from flask import Flask, request, url_for, send_from_directory
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import np_utils
from keras.utils import plot_model
from keras.preprocessing import image as kimage
from keras.applications.imagenet_utils import preprocess_input
def num_recognition(weights_path=None):
    num_classes = 10
    img_width, img_height = 28, 28
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 1)
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if weights_path:
        model.load_weights(weights_path)
    return model

model = num_recognition('6-layers-CNN.h5')

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()+"/nums"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>数字图片上传</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=file>
         <input type=submit value=上传>
    </form>
    '''

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename =file.filename
            kimagePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)

            img = kimage.load_img(kimagePath, grayscale=True, target_size=(28, 28))
            x = kimage.img_to_array(img)
            x = x.reshape((1,) + x.shape)
            proba = model.predict_proba(x, verbose=0)
            result = model.predict_classes(x, verbose=0)
            print('识别为：')
            # print(proba[0])
            print(result)
            r = str(result[0])
            return html + '<br><img src=' + file_url + ' width="200px" height="200px"><br><p> 该图片数字为：'+ r + "</p>"
    return html

if __name__ == '__main__':
    app.run(host="10.108.104.192",
            port = 5000)