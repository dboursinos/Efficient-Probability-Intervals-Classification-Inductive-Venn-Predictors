# -*- coding: utf-8 -*-
from __future__ import division
import keras
from keras import backend as K
import numpy as np
import random
import numpy.random as rng

def __randint_unequal(lower, upper):
        int_1 = random.randint(lower, upper)
        int_2 = random.randint(lower, upper)
        while int_1 == int_2:
            int_1 = random.randint(lower, upper)
            int_2 = random.randint(lower, upper)
        return int_1, int_2

def class_separation(y_train):
    class_idxs=[]
    for data_class in sorted(set(y_train)):
        class_idxs.append(np.where((y_train == data_class))[0])
    return class_idxs

def euclidean_loss(y_true, y_pred):
    loss=y_true*K.square(y_pred)+(1-y_true)*K.square(K.maximum(5-y_pred,0))
    return loss

def get_batch(batch_size,x_train,y_train,idxs_per_class):
    n_classes=len(idxs_per_class)

    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes,size=(batch_size,),replace=True)
    # print("categories")
    # print(categories)
    
    # pairs1 has the anchors while pairs2 is either positive or negative
    pairs1=[]
    pairs2=[]
    
    # initialize vector for the targets
    targets=np.zeros((batch_size,),dtype='float')
    
    # make lower half of it '1's, so 2nd half of batch has same class
    targets[batch_size//2:] = 1.0
    for i in range(batch_size):
        category = categories[i]
        if i>=batch_size//2: #positive
            idx = rng.choice(len(idxs_per_class[category]),size=(2,),replace=False)
            # print(idx[0],idx[1],y_train[idxs_per_class[category][idx[0]]],y_train[idxs_per_class[category][idx[1]]])
            pairs1.append(x_train[idxs_per_class[category][idx[0]]])
            pairs2.append(x_train[idxs_per_class[category][idx[1]]])
        else: #negative
            category2=(category+rng.randint(1,n_classes-1))%n_classes #pick from a different class
            # category2=(category+1)%2
            idx1 = rng.randint(0, len(idxs_per_class[category]))
            idx2 = rng.randint(0, len(idxs_per_class[category2]))
            # print(idx1,idx2,y_train[idxs_per_class[category][idx1]],y_train[idxs_per_class[category2][idx2]])
            pairs1.append(x_train[idxs_per_class[category][idx1]])
            pairs2.append(x_train[idxs_per_class[category2][idx2]])
    
    return pairs1, pairs2, targets
