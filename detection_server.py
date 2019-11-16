import cv2
import numpy as np
import os
from flask import Flask, request, redirect, url_for
from edge_enhancement import enhance_edge
import tensorflow as tf
from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import glob
from numpy import genfromtxt
from fr_utils import *
from inception_network import *
from sign_functions import *
from keras.models import load_model
import sys 

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])
DATABASE_PATH = "images/"

_model = None
_sess = None
_graph = None

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/add', methods=['POST'])
def add_new_signature():
    if 'image' not in request.files:
        print('No file part')
        return '', 400

    file = request.files['image']
    if file.filename == '':
        print('No selected file')
        return '', 400

    if 'name' not in request.form:
        print('No name provided')
        return '', 400

    name = request.form.get('name')

    if os.path.isfile(DATABASE_PATH + name+".jpg"):
        print('Signature exists already.')
        return '', 409

    if file and allowed_file(file.filename):
        filestr = file.read()
        npimg = np.fromstring(filestr, np.uint8)
        img = enhance_edge(cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED))
        cv2.imwrite(DATABASE_PATH + name + ".jpg", img)
    return '', 200

@app.route('/predict', methods=['POST'])
def predict():
    global _model
    if 'image' not in request.files:
        print('No file part')
        return '', 400

    file = request.files['image']
    if file.filename == '':
        print('No selected file')
        return '', 400

    if 'name' not in request.form:
        print('No name provided')
        return '', 400

    name = request.form.get('name')

    if not os.path.isfile(DATABASE_PATH + name+".jpg"):
        print('Signature does not exist.')
        return '', 404

    if file and allowed_file(file.filename):
        filestr = file.read()
        npimg = np.fromstring(filestr, np.uint8)
        img = enhance_edge(cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED))
        cv2.imwrite('temp.jpg', img)
        database = prepare_database(_model)
        owner = recognise_sign(cv2.imread('temp.jpg', 1), database, _model)
        os.remove('temp.jpg')
        print(owner)
        if(owner == 0 or owner != name):
            print('Signature does not match')
            return '', 404
        else:
            print('Signature found: ' + owner)
            return '', 200

def triplet_loss_function(y_true,y_pred,alpha = 0.3):
	anchor = y_pred[0]
	positive = y_pred[1]
	negative = y_pred[2]
	pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
	neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
	basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
	loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
	return loss

def load_session():
    global _sess
    _sess = tf.Session()

def load_graph():
    global _graph
    _graph = tf.get_default_graph()

def load_model():
    global _model
    _model = model(input_shape = (3,96,96))
    _model.compile(optimizer = 'adam', loss = triplet_loss_function, metrics = ['accuracy'])
    load_weights_from_FaceNet(_model)

if __name__ == '__main__':
    load_session()
    load_graph()
    load_model()

    x = np.random.rand(96, 96, 3)
    y = np.random.rand(96, 96, 3)

    try:
        database = prepare_database(_model)
        face = recognise_sign(x, database, _model)
    except:
        load_model()

    '''print('compiling Model.....')
   
    print('model compile sucessful')
    print('loading weights into model, this might take sometime sir!')

    print('loading weights sequence complete sir!')'''

    app.run(host= '0.0.0.0')
