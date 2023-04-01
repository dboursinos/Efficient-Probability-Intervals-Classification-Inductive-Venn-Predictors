import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import sys
import pickle
import matplotlib.pyplot as plt
import config

sys.setrecursionlimit(40000)
C = config.Config()

with open(C.preprocessed_data, "rb") as f:
    data = pickle.load(f)

with open(C.train_embeddings_path, "rb") as f:
    train_embeds = pickle.load(f)
with open(C.calibration_embeddings_path, "rb") as f:
    calibration_embeds = pickle.load(f)

# KNN
neigh = NearestNeighbors(n_neighbors=C.knn_neighbors, algorithm='kd_tree', metric='euclidean')
neigh.fit(train_embeds)
indices=neigh.kneighbors(calibration_embeds, return_distance=False)
calibration_nc=np.empty(len(data['y_validation']))
for i in range(len(data['y_validation'])):
    calibration_nc[i]=np.count_nonzero(data['y_train'][indices[i]]!=data['y_validation'][i])

with open(C.calibration_nc_scores['knn'], "wb") as f:
    pickle.dump((calibration_nc,neigh), f)


# Nearest Centroid
centroids=np.empty((C.num_classes,C.embeddings_size))
for i in range(C.num_classes):
    centroids[i]=np.mean(train_embeds[data['y_train']==i],axis=0)

calibration_nc=np.empty(len(data['y_validation']))
temp_distances=np.zeros(C.num_classes)
for i in range(len(data['y_validation'])):
    for j in range(C.num_classes):
        temp_distances[j]=np.linalg.norm(calibration_embeds[i]-centroids[j])
    calibration_nc[i]=temp_distances[data['y_validation'][i]]/np.min(temp_distances[np.arange(len(temp_distances))!=data['y_validation'][i]])


with open(C.calibration_nc_scores['nearest_centroid'], "wb") as f:
    pickle.dump((calibration_nc,centroids), f)


# 1NN
in_class_knn=[]
out_class_knn=[]   
calibration_nc=np.empty(len(data['y_validation']))

for i in range(C.num_classes):
    knn=NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(train_embeds[data['y_train']==i])
    in_class_knn.append(knn)
    knn=NearestNeighbors(n_neighbors=1, algorithm='auto', metric='euclidean').fit(train_embeds[data['y_train']!=i])
    out_class_knn.append(knn)

calibration_nc=np.empty(len(data['y_validation']))
for i in range(len(data['y_validation'])):
    dist_in,_ = in_class_knn[data['y_validation'][i]].kneighbors(calibration_embeds[i].reshape(1,-1))
    dist_out,_ = out_class_knn[data['y_validation'][i]].kneighbors(calibration_embeds[i].reshape(1,-1))
    calibration_nc[i]=dist_in/dist_out
    

with open(C.calibration_nc_scores['1nn'], "wb") as f:
    pickle.dump((calibration_nc,in_class_knn,out_class_knn), f)
