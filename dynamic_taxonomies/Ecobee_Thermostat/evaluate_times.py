import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import sys
import pickle
import time
import csv
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

try:
    print('loading classifier model from {}'.format(C.classifier_model_path))
    classifier_model=load_model(C.classifier_model_path)
except:
    print('Could not load the classifier model. Check the given path')

with open(C.data_path, "rb") as f:
    data = pickle.load(f)

times={}

for taxon in ['v1','v2','v3','v4']:
    print(taxon)
    with open(C.taxonomies_paths[taxon], "rb") as f:
        category_distributions = pickle.load(f)

    start_time = time.process_time()
    for test_in in data['x_test']:
        test_prob=classifier_model.predict(np.expand_dims(test_in,axis=0))[0]
        
        
        if taxon=='v2':
            test_prediction=np.argmax(test_prob)
            category=test_prediction*2+(test_prob>0.75)
        elif taxon=='v3':
            test_prediction=np.argmax(test_prob)
            sorted_probs=np.sort(test_prob)
            second_highest_probs=sorted_probs[-2]
            category=test_prediction*2+(second_highest_probs>0.25)
        elif taxon=='v4':
            test_prediction=np.argmax(test_prob)
            sorted_probs=np.sort(test_prob)
            second_highest_probs=sorted_probs[-2]
            category=test_prediction*2+(test_prob-second_highest_probs>0.5)
        elif taxon=='v1':
            test_prediction=np.argmax(test_prob)
            category=test_prediction

        samples_in_category=np.sum(category_distributions[category,:])
        temp_distribution_min=category_distributions[category,:]/(samples_in_category+1)
        temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))

    end_time = time.process_time()
    times[taxon]=(end_time-start_time)/len(data['x_test'])



for taxon in ['knn_v1','knn_v2','nc_v1','nc_v2']:
    print(taxon)
    if taxon=='knn_v1':
        with open(C.taxonomies_paths[taxon], "rb") as f:
            neigh,category_distributions = pickle.load(f)

        start_time = time.process_time()

        for test_in in data['x_test']:
            test_embedding=siamese_model.predict(np.expand_dims(test_in,axis=0))[0]
            category=neigh.predict(test_embedding.reshape(1, -1))
            samples_in_category=np.sum(category_distributions[category,:])
            temp_distribution_min=category_distributions[category,:]/(samples_in_category+1)
            temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))

        end_time = time.process_time()
        times[taxon]=(end_time-start_time)/len(data['x_test'])

    if taxon=='knn_v2':
        with open(C.taxonomies_paths[taxon], "rb") as f:
            neigh,category_distributions = pickle.load(f)

        start_time = time.process_time()

        for test_in in data['x_test']:
            test_embedding=siamese_model.predict(np.expand_dims(test_in,axis=0))[0]
            test_neighbor_indexes=neigh.kneighbors(test_embedding.reshape(1, -1),return_distance=False)
            test_neighbor_labels=data['y_train'][test_neighbor_indexes]
            category=neigh.predict(test_embedding.reshape(1, -1))

            samples_in_category=np.sum(category_distributions[category,:])
            temp_distribution_min=category_distributions[category,:]/(samples_in_category+1)
            temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))

        end_time = time.process_time()
        times[taxon]=(end_time-start_time)/len(data['x_test'])

    if taxon=='nc_v1':
        with open(C.taxonomies_paths[taxon], "rb") as f:
            centroids,category_distributions = pickle.load(f)

        temp_distances=np.zeros(C.num_classes)
        start_time = time.process_time()

        for test_in in data['x_test']:
            test_embedding=siamese_model.predict(np.expand_dims(test_in,axis=0))[0]
            for j in range(C.num_classes):
                temp_distances[j]=np.linalg.norm(test_embedding-centroids[j])
            category=np.argmin(temp_distances)
            
            samples_in_category=np.sum(category_distributions[category,:])
            temp_distribution_min=category_distributions[category,:]/(samples_in_category+1)
            temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))

        end_time = time.process_time()
        times[taxon]=(end_time-start_time)/len(data['x_test'])

    if taxon=='nc_v2':
        with open(C.taxonomies_paths[taxon], "rb") as f:
            centroids,category_distributions = pickle.load(f)

        temp_distances=np.zeros(C.num_classes)
        start_time = time.process_time()

        for test_in in data['x_test']:
            test_embedding=siamese_model.predict(np.expand_dims(test_in,axis=0))[0]
            for j in range(C.num_classes):
                temp_distances[j]=np.linalg.norm(test_embedding-centroids[j])
            category=2*np.argmin(temp_distances)+int(np.min(temp_distances)>0.08)
            
            samples_in_category=np.sum(category_distributions[category,:])
            temp_distribution_min=category_distributions[category,:]/(samples_in_category+1)
            temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))

        end_time = time.process_time()
        times[taxon]=(end_time-start_time)/len(data['x_test'])

with open(C.execution_times_path, 'w', newline='') as csvfile:
    fieldnames = ['v1','v2','v3','v4','knn_v1','knn_v2','nc_v1','nc_v2']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(times)
