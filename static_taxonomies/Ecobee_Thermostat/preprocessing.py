from PIL import Image,ImageOps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pickle
import numpy as np
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math
import config

C=config.Config()

np.random.seed(42)

train_data = [] # images
train_labels = [] # corresponding labels
test_data = []
test_labels = []

benign_data=np.loadtxt(open(C.benign_data_path, "rb"), delimiter=",",skiprows=1)
min_max_scaler = MinMaxScaler().fit(benign_data)
merged_data = min_max_scaler.transform(benign_data)
labels=np.zeros(merged_data.shape[0],dtype='int')
# del benign_data

for a_type in C.mirai_attacks:
	attack_data_scaled=min_max_scaler.transform(np.loadtxt(open(C.mirai_data_path+C.mirai_attacks[a_type][0], "rb"), delimiter=",",skiprows=1))
	if attack_data_scaled.shape[0]>benign_data.shape[0]:
		rnd=np.random.choice(attack_data_scaled.shape[0],benign_data.shape[0],replace=False)
		attack_data_scaled=attack_data_scaled[rnd,:]
	merged_data=np.vstack((merged_data,attack_data_scaled))
	labels=np.hstack((labels,C.mirai_attacks[a_type][1]*np.ones(attack_data_scaled.shape[0],dtype='int')))

for a_type in C.gafgyt_attacks:
	attack_data_scaled=min_max_scaler.transform(np.loadtxt(open(C.gafgyt_data_path+C.gafgyt_attacks[a_type][0], "rb"), delimiter=",",skiprows=1))
	if attack_data_scaled.shape[0]>benign_data.shape[0]:
		rnd=np.random.choice(attack_data_scaled.shape[0],benign_data.shape[0],replace=False)
		attack_data_scaled=attack_data_scaled[rnd,:]
	merged_data=np.vstack((merged_data,attack_data_scaled))
	labels=np.hstack((labels,C.gafgyt_attacks[a_type][1]*np.ones(attack_data_scaled.shape[0],dtype='int')))

train_data, test_data, train_labels, test_labels = train_test_split(merged_data, labels, test_size=0.1, random_state=42)
train_data, validation_data, train_labels, validation_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)


data={'x_train':train_data,
'y_train': train_labels,
'x_validation' : validation_data,
'y_validation' : validation_labels,
'x_test': test_data,
'y_test': test_labels}

with open(C.data_path, 'wb') as f:
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

fig.savefig("dataset_class_distribution.png")
