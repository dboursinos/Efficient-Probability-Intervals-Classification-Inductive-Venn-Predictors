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
    print('loading siamese model from {}'.format(C.siamese_model_path))
    siamese_model=load_model(C.siamese_model_path)
except:
    print('Could not load the siamese model. Check the given path')

try:
    print('loading classifier model from {}'.format(C.classifier_model_path))
    classifier_model=load_model(C.classifier_model_path)
except:
    print('Could not load the siamese model. Check the given path')

with open(C.data_path, "rb") as f:
    data = pickle.load(f)

with open(C.siamese_embeddings_paths['train'], "rb") as f:
    train_embeds = pickle.load(f)
with open(C.siamese_embeddings_paths['validation'], "rb") as f:
    calibration_embeds = pickle.load(f)

with open(C.classifier_probs_paths['validation'], "rb") as f:
    calibration_probs = pickle.load(f)


baseline_taxonomies={
    'v1':taxonomies_calibration.nn_v1,
    'v2':taxonomies_calibration.nn_v2,
    'v3':taxonomies_calibration.nn_v3,
    'v4':taxonomies_calibration.nn_v4
}

proposed_taxonomies={
    'knn_v1':taxonomies_calibration.knn_v1,
    'knn_v2':taxonomies_calibration.knn_v2,
    'nc_v1':taxonomies_calibration.nc_v1,
    'nc_v2':taxonomies_calibration.nc_v2
}

for key, value in baseline_taxonomies.items():
    print(key)
    value(data['y_validation'],calibration_probs,C)

for key, value in proposed_taxonomies.items():
    print(key)
    value(train_embeds,data['y_train'],calibration_embeds,data['y_validation'],C)

