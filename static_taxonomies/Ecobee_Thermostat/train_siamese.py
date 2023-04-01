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
from keras.utils import generic_utils
from keras.callbacks import TensorBoard
import config
import helper
from simple_FF import create_base_model
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

with open(C.data_path, "rb") as f:
    data = pickle.load(f)

in_dims=data['x_train'].shape[1:]
fc_model=create_base_model(in_dims,C.embeddings_size,C.num_classes)
base_model=Model(inputs=fc_model.input,outputs=fc_model.get_layer('embedding').output)
base_model.summary()

input_a = Input(shape=(in_dims[0],))
input_b = Input(shape=(in_dims[0],))
processed_a = base_model(input_a)
processed_b = base_model(input_b)
l2_distance_layer = Lambda(
            lambda tensors: K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True))
l2_distance = l2_distance_layer([processed_a, processed_b])
siamese_network=Model([input_a, input_b], l2_distance)

siamese_network.compile(loss=helper.euclidean_loss, optimizer=keras.optimizers.adam(lr=0.0001))
siamese_network.summary()

idxs_per_class=helper.class_separation(data['y_train'])
with open(C.idxs_class_train_path, "wb") as f:
    pickle.dump(idxs_per_class, f)

batch_size=128
num_epochs=2000
epoch_length=200
callbacks=[]

log_path = './logs'
if not os.path.isdir(log_path):
    os.mkdir(log_path)

callback = TensorBoard(log_path)
callback.set_model(siamese_network)

best_loss = np.Inf
train_step = 0
losses = np.zeros(epoch_length)
for epoch_num in range(num_epochs):
    print('Epoch {}/{}'.format(epoch_num+1,num_epochs))
    progbar = generic_utils.Progbar(epoch_length)   # keras progress bar
    iter_num = 0
    start_time = time.time()
    for batch_num in range(epoch_length):
        inputs1,inputs2,targets=helper.get_batch(batch_size,data['x_train'],data['y_train'],idxs_per_class)
        loss = siamese_network.train_on_batch([inputs1, inputs2], targets)
        write_log(callback, ['loss'], [loss, train_step],train_step)
        losses[iter_num] = loss
        iter_num+=1
        train_step += 1
        progbar.update(iter_num, [('loss', np.mean(losses[:iter_num]))])

        if iter_num == epoch_length:
            epoch_loss = np.mean(losses)
            write_log(callback,
                        ['Elapsed_time', 'mean_loss'],
                        [time.time() - start_time, epoch_loss],
                        epoch_num)
            if epoch_loss < best_loss:
                print('Total loss decreased from {} to {}, saving weights'.format(best_loss,epoch_loss))
                best_loss = epoch_loss
                base_model.save(C.siamese_model_path)
