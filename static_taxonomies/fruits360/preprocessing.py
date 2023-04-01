from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.datasets import load_files
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from collections import Counter
import pickle
import config

C = config.Config()


def load_dataset(data_path):
    print(1)
    data_loading = load_files(data_path)
    files_add = np.array(data_loading['filenames'])
    targets_fruits = np.array(data_loading['target'])
    target_labels_fruits = np.array(data_loading['target_names'])
    return files_add,targets_fruits,target_labels_fruits

def convert_image_to_array_form(files):
    images_array=[]
    for file in files:
        images_array.append(img_to_array(load_img(file)))
    return images_array
    
x_train, y_train,target_labels = load_dataset(C.train_dir)
x_test, y_test,_ = load_dataset(C.test_dir)

no_of_classes = len(np.unique(y_train))
class_mapping={}
for i,cl in enumerate(target_labels):
    if cl not in class_mapping:
        class_mapping[cl]=y_train[i]
print(class_mapping)
with open(C.class_mapping_path, "wb") as f:
    pickle.dump(class_mapping, f)

x_test,x_valid = x_test[7000:],x_test[:7000]
y_test,y_valid = y_test[7000:],y_test[:7000]
print('Vaildation X : ',x_valid.shape)
print('Vaildation y :',y_valid.shape)
print('Test X : ',x_test.shape)
print('Test y : ',y_test.shape)

x_train = np.array(convert_image_to_array_form(x_train))
print('Training set shape : ',x_train.shape)

x_valid = np.array(convert_image_to_array_form(x_valid))
print('Validation set shape : ',x_valid.shape)

x_test = np.array(convert_image_to_array_form(x_test))
print('Test set shape : ',x_test.shape)

print('1st training image shape ',x_train[0].shape)

labels, values = zip(*Counter(y_train).items())
indexes = np.arange(len(labels))
width = 1
fig = plt.figure(figsize=(50, 6))
plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.savefig('train_hist')

x_train = x_train.astype('float32')/255
x_valid = x_valid.astype('float32')/255
x_test = x_test.astype('float32')/255


print('Train data shape: ', x_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', x_valid.shape)
print('Validation labels shape: ', y_valid.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)

data={'x_train':x_train,
	'y_train':y_train,
	'x_val':x_valid,
	'y_val':y_valid,
	'x_test':x_test,
	'y_test':y_test}

with open(C.data_path, "wb") as f:
    pickle.dump(data, f, protocol=4)

print(x_train[15])
print(y_train[15])

fig = plt.figure(figsize=(12, 12))
plt.imshow(x_train[15])
plt.savefig('example')
