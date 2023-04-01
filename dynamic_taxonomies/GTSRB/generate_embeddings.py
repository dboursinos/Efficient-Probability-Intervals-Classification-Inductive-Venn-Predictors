import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import sys
import pickle
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

siamese_model.summary()

with open(C.preprocessed_data, "rb") as f:
    data = pickle.load(f)

train_embeds=siamese_model.predict(data['x_train'])
calibration_embeds=siamese_model.predict(data['x_validation'])
test_embeds=siamese_model.predict(data['x_test'])

with open(C.train_embeddings_path, "wb") as f:
    pickle.dump(train_embeds, f)

with open(C.calibration_embeddings_path, "wb") as f:
    pickle.dump(calibration_embeds, f)

with open(C.test_embeddings_path, "wb") as f:
    pickle.dump(test_embeds, f)


