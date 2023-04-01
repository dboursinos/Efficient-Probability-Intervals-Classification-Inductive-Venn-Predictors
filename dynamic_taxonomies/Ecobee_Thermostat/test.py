import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import csv
import pickle
import config
import evaluation_metrics
import taxonomies_test

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.compat.v1.Session(config=tf_config)

K.set_learning_phase(0)

sys.setrecursionlimit(40000)
C = config.Config()


with open(C.data_path, "rb") as f:
    data = pickle.load(f)

with open(C.siamese_embeddings_paths['test'], "rb") as f:
    test_embeds = pickle.load(f)
   

proposed_taxonomies={
    'knn_v1':taxonomies_test.knn_v1,
    'knn_v2':taxonomies_test.knn_v2,
    'nc_v1':taxonomies_test.nc_v1,
    'nc_v2':taxonomies_test.nc_v2
}


for key, value in proposed_taxonomies.items():
    print(key)
    if key!='knn_v2':
        value(test_embeds,data,C)
    else:
        value(test_embeds,data,C)
