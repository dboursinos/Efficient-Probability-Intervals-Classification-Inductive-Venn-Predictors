import time
import numpy as np
import pickle
import os
import sys

from collections import Counter
import tensorflow as tf
import keras
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop, adam
from keras.layers import Input,Lambda, Dense
from keras.models import Model, load_model
from keras.losses import categorical_crossentropy
from keras.utils import generic_utils, to_categorical
from keras.callbacks import TensorBoard,ModelCheckpoint,CSVLogger,EarlyStopping
import config
from siamese_model import helper, vgg, mobilenet
from sklearn import preprocessing

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.compat.v1.Session(config=tf_config)

sys.setrecursionlimit(40000)
np.random.seed(42)
C = config.Config()

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

with open(C.preprocessed_data, "rb") as f:
    data = pickle.load(f)

in_dims=data['x_train'].shape[1:]

cnn_model=mobilenet.create_base_model(in_dims,C.alpha,C.embeddings_size,C.num_classes)
cnn_model.compile(loss=categorical_crossentropy,
                 optimizer=Adam(),
                 metrics=['accuracy'])
cnn_model.summary()

callbacks = [CSVLogger(C.classifier_train_loss_path,separator=' '),
				 EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)]
cnn_model.fit(data['x_train'],
        to_categorical(data['y_train']),
        validation_data=(data['x_validation'],to_categorical(data['y_validation'])),
        batch_size=64,
        epochs=1000,
        callbacks=callbacks,
        verbose=1)
cnn_model.save(C.classifier_model_path)
