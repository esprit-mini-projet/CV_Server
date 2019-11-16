import cv2
import numpy as np
import glob
import os
from fr_utils import img_to_encoding

def recognise_sign(image, database, model):
    encoding = img_to_encoding(image, model)
    identity = None
    min_dist = 100
    for (name, db_enc) in database.items():
        
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' %(name, dist))
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.35:
        print('cant recognisethe face', 2)
        return str(0)
    else:
        return str(identity)

def prepare_database(model):

    database = {}
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = img_to_encoding(cv2.imread(file, 1), model)

    return database