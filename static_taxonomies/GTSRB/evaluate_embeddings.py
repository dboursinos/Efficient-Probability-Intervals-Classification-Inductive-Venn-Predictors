import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import sys
import pickle
import plots
import plot_functions
import config

sys.setrecursionlimit(40000)
C = config.Config()

with open(C.preprocessed_data, "rb") as f:
    data = pickle.load(f)

with open(C.train_embeddings_path, "rb") as f:
    train_embeds = pickle.load(f)
with open(C.calibration_embeddings_path, "rb") as f:
    calibration_embeds = pickle.load(f)
with open(C.test_embeddings_path, "rb") as f:
    test_embeds = pickle.load(f)

plot_functions.evaluate_siamese(train_embeds,data['y_train'],calibration_embeds,data['y_validation'],test_embeds,data['y_test'],C)