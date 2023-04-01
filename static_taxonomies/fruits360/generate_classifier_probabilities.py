import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import sys
import pickle
import taxonomies_calibration
import config

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.compat.v1.Session(config=tf_config)

K.set_learning_phase(0)

sys.setrecursionlimit(40000)
C = config.Config()

try:
    print('loading classifier model from {}'.format(C.classifier_model_path))
    classifier_model=load_model(C.classifier_model_path)
except:
    print('Could not load the classifier model. Check the given path')

with open(C.data_path, "rb") as f:
    data = pickle.load(f)    

train_probs=classifier_model.predict(data['x_train'])
calibration_probs=classifier_model.predict(data['x_val'])
test_probs=classifier_model.predict(data['x_test'])

with open(C.train_probs_path, "wb") as f:
    pickle.dump(train_probs, f)

with open(C.calibration_probs_path, "wb") as f:
    pickle.dump(calibration_probs, f)

with open(C.test_probs_path, "wb") as f:
    pickle.dump(test_probs, f)
