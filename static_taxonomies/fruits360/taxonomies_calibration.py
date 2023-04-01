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


def nn_v1(y_cal,calibration_probabilities,C):
    taxon='v1'
    calibration_predictions=np.argmax(calibration_probabilities,axis=1)
    print(np.sum(calibration_predictions==y_cal)/len(y_cal))
    category_distributions=np.zeros((C.num_classes,C.num_classes),dtype='int')
    for i in range(len(calibration_predictions)):
        category_distributions[calibration_predictions[i],y_cal[i]]+=1

    with open(C.taxonomies_paths[taxon], "wb") as f:
        pickle.dump(category_distributions, f)

def nn_v2(y_cal,calibration_probabilities,C):
    taxon='v2'
    calibration_predictions=np.argmax(calibration_probabilities,axis=1)
    calibration_prediction_probs=np.max(calibration_probabilities,axis=1)

    category_distributions=np.zeros((2*C.num_classes,C.num_classes),dtype='int')
    for i in range(len(calibration_predictions)):
        category_distributions[calibration_predictions[i]*2+(calibration_prediction_probs[i]>0.75),y_cal[i]]+=1
    with open(C.taxonomies_paths[taxon], "wb") as f:
        pickle.dump(category_distributions, f)

def nn_v3(y_cal,calibration_probabilities,C):
    taxon='v3'
    calibration_predictions=np.argmax(calibration_probabilities,axis=1)
    calibration_prediction_probs=np.max(calibration_probabilities,axis=1)
    sorted_probs=np.sort(calibration_probabilities,axis=1)
    second_highest_probs=sorted_probs[:,-2]

    category_distributions=np.zeros((2*C.num_classes,C.num_classes),dtype='int')
    for i in range(len(calibration_predictions)):
        category_distributions[calibration_predictions[i]*2+(second_highest_probs[i]>0.25),y_cal[i]]+=1
    with open(C.taxonomies_paths[taxon], "wb") as f:
        pickle.dump(category_distributions, f)

def nn_v4(y_cal,calibration_probabilities,C):
    taxon='v4'
    calibration_predictions=np.argmax(calibration_probabilities,axis=1)
    calibration_prediction_probs=np.max(calibration_probabilities,axis=1)
    sorted_probs=np.sort(calibration_probabilities,axis=1)
    second_highest_probs=sorted_probs[:,-2]

    category_distributions=np.zeros((2*C.num_classes,C.num_classes),dtype='int')
    for i in range(len(calibration_predictions)):
        category_distributions[calibration_predictions[i]*2+(calibration_prediction_probs[i]-second_highest_probs[i]>0.5),y_cal[i]]+=1
    with open(C.taxonomies_paths[taxon], "wb") as f:
        pickle.dump(category_distributions, f)


def knn_v1(train_embeds,y_train,calibration_embeds,y_cal,C):
    taxon='knn_v1'
    neigh = KNeighborsClassifier(n_neighbors=C.k_neighbors, metric="euclidean")
    neigh.fit(train_embeds, y_train)
    calibration_predictions=neigh.predict(calibration_embeds)

    category_distributions=np.zeros((C.num_classes,C.num_classes),dtype='int')
    for i in range(C.num_classes):
        for j in range(C.num_classes):
            category_distributions[i,j]=np.count_nonzero(y_cal[calibration_predictions==i]==j)

    with open(C.taxonomies_paths[taxon], "wb") as f:
        pickle.dump((neigh,category_distributions), f)

def knn_v2(train_embeds,y_train,calibration_embeds,y_cal,C):
    taxon='knn_v2'
    neigh = KNeighborsClassifier(n_neighbors=C.k_neighbors, metric="euclidean")
    neigh.fit(train_embeds, y_train)
    calibration_predictions=neigh.predict(calibration_embeds)
    neigh_ind=neigh.kneighbors(calibration_embeds,return_distance=False)
    neigh_labels=y_train[neigh_ind]

    num_categories=(C.k_neighbors//2+1)*C.num_classes
    category_distributions=np.zeros((num_categories,C.num_classes),dtype='int')
    for i in range(len(calibration_embeds)):
        category_distributions[calibration_predictions[i]*(C.k_neighbors//2+1)+np.count_nonzero(neigh_labels[i]!=calibration_predictions[i])][y_cal[i]]+=1

    with open(C.taxonomies_paths[taxon], "wb") as f:
        pickle.dump((neigh,category_distributions), f)

def nc_v1(train_embeds,y_train,calibration_embeds,y_cal,C):
    taxon='nc_v1'
    centroids=np.empty((C.num_classes,C.embeddings_size))
    for i in range(C.num_classes):
        centroids[i]=np.mean(train_embeds[y_train==i],axis=0)
    
    category_distributions=np.zeros((C.num_classes,C.num_classes),dtype='int')
    temp_distances=np.zeros(C.num_classes)
    for i in range(len(calibration_embeds)):
        for j in range(C.num_classes):
            temp_distances[j]=np.linalg.norm(calibration_embeds[i]-centroids[j])
        category_distributions[np.argmin(temp_distances)][y_cal[i]]+=1

    with open(C.taxonomies_paths[taxon], "wb") as f:
        pickle.dump((centroids,category_distributions), f)

def nc_v2(train_embeds,y_train,calibration_embeds,y_cal,C):
    taxon='nc_v2'
    centroids=np.empty((C.num_classes,C.embeddings_size))
    for i in range(C.num_classes):
        centroids[i]=np.mean(train_embeds[y_train==i],axis=0)

    category_distributions=np.zeros((2*C.num_classes,C.num_classes),dtype='int')
    temp_distances=np.zeros(C.num_classes)
    for i in range(len(calibration_embeds)):
        for j in range(C.num_classes):
            temp_distances[j]=np.linalg.norm(calibration_embeds[i]-centroids[j])
        pred=np.argmin(temp_distances)
        dist=np.min(temp_distances)
        category_distributions[2*pred+int(dist>0.08)][y_cal[i]]+=1

    with open(C.taxonomies_paths[taxon], "wb") as f:
        pickle.dump((centroids,category_distributions), f)

def nn2_nc2(train_embeds,y_train,x_cal,calibration_embeds,y_cal,classifier_model,C):
    centroids=np.empty((C.num_classes,C.embeddings_size))
    for i in range(C.num_classes):
        centroids[i]=np.mean(train_embeds[y_train==i],axis=0)

    calibration_probabilities=classifier_model.predict(x_cal)
    calibration_predictions=np.argmax(calibration_probabilities,axis=1)
    calibration_prediction_probs=np.max(calibration_probabilities,axis=1)

    num_nc2_categories=2*C.num_classes
    num_nn2_categories=2*C.num_classes
    category_distributions=np.zeros((num_nc2_categories*num_nn2_categories,C.num_classes),dtype='int')

    temp_distances=np.zeros(C.num_classes)
    for i in range(len(calibration_embeds)):
        nn2_c=calibration_predictions[i]*2+(calibration_prediction_probs[i]>0.75)
        for j in range(C.num_classes):
            temp_distances[j]=np.linalg.norm(calibration_embeds[i]-centroids[j])
        pred=np.argmin(temp_distances)
        dist=np.min(temp_distances)
        nc2_c=2*pred+int(dist>0.08)
        category_distributions[nc2_c*num_nn2_categories+nn2_c][y_cal[i]]+=1

    with open(C.nn2_nc2_calibration_results_path, "wb") as f:
        pickle.dump((centroids,category_distributions), f)
        
