from PIL import Image,ImageOps
from sklearn.model_selection import train_test_split
import config
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

C = config.Config()

num_classes=43
input_side_size=96

train_images = [] # images
train_labels = [] # corresponding labels
test_images = []
test_labels = []

for c in range(num_classes):
    prefix = C.training_set_path + '/' + format(c, '05d') + '/' # subdirectory for class
    gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader) # skip header
    for row in gtReader:
        img=Image.open(prefix + row[0]) # the 1th column is the filename
        img=img.resize([C.input_side_size,C.input_side_size],Image.ANTIALIAS)
        img_rgb=np.array(img)
        train_images.append(img_rgb)
        train_labels.append(row[7]) # the 8th column is the label
    gtFile.close()

train_images=np.array(train_images,dtype='float32')/255.
train_labels=np.array(train_labels,dtype='int')

train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=C.test_ratio, random_state=42)
train_images, calibration_images, train_labels, calibration_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

data={'x_train':train_images,
'y_train': train_labels,
'x_validation' : calibration_images,
'y_validation' : calibration_labels,
'x_test': test_images,
'y_test': test_labels}

with open(C.preprocessed_data, 'wb') as f:
    pickle.dump(data, f)

fig=plt.figure()
fig.set_size_inches(15,7)
label_counter=Counter(data['y_train'])
ax1=fig.add_subplot(131)
ax1.set_ylabel('Count',color='k',fontsize=16)
ax1.set_xlabel('Classes',color='k',fontsize=16)
ax1.bar(label_counter.keys(), label_counter.values())
ax1.grid(False)
ax1.set_title("Training Samples")

label_counter=Counter(data['y_validation'])
ax2=fig.add_subplot(132)
ax2.set_ylabel('Count',color='k',fontsize=16)
ax2.set_xlabel('Classes',color='k',fontsize=16)
ax2.bar(label_counter.keys(), label_counter.values())
ax2.grid(False)
ax2.set_title("Validation Samples")

label_counter=Counter(data['y_test'])
ax3=fig.add_subplot(133)
ax3.set_ylabel('Count',color='k',fontsize=16)
ax3.set_xlabel('Classes',color='k',fontsize=16)
ax3.bar(label_counter.keys(), label_counter.values())
ax3.grid(False)
ax3.set_title("Testing Samples")

fig.savefig(C.class_sizes_plot_path)
