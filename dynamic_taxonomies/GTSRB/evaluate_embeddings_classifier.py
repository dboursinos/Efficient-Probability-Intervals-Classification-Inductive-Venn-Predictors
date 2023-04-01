import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import sys
import pickle
import plot_functions
import config

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.compat.v1.Session(config=tf_config)

K.set_learning_phase(0)

sys.setrecursionlimit(40000)
C = config.Config()

with open(C.preprocessed_data, "rb") as f:
    data = pickle.load(f)

with open(C.classifier_embeddings_paths['train'], "rb") as f:
    train_embeds = pickle.load(f)
with open(C.classifier_embeddings_paths['validation'], "rb") as f:
    calibration_embeds = pickle.load(f)
with open(C.classifier_embeddings_paths['test'], "rb") as f:
    test_embeds = pickle.load(f)

plot_functions.evaluate_siamese(train_embeds,data['y_train'],calibration_embeds,data['y_validation'],test_embeds,data['y_test'],C)
