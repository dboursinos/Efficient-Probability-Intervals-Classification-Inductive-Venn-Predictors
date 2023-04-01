# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import scikitplot as skplt
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np

def scatter(x, labels_str, filename, subtitle=None):
    # We choose a color palette with seaborn.
    le=LabelEncoder()
    labels = le.fit_transform(labels_str)
    palette = np.array(sns.color_palette("hls", np.max(labels)+1))

    # We create a scatter plot.
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=5,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(np.max(labels)+1):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=14)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
        
    if subtitle != None:
        plt.suptitle(subtitle)
        
    plt.show()
    plt.savefig(filename)
    plt.close(fig)
    
def confusion_matrix(ground_truth_labels,predicted_labels,title,fsize,path):
    skplt.metrics.plot_confusion_matrix(ground_truth_labels, predicted_labels,title=title,figsize=fsize)
    plt.savefig(path)

def evaluate_siamese(train_embeds,train_labels,calibration_embeds,calibration_labels,test_embeds,test_labels,config):
    neigh = KNeighborsClassifier(n_neighbors=10, metric="euclidean", algorithm='kd_tree')
    neigh.fit(train_embeds, train_labels)
    train_predictions=neigh.predict(train_embeds)
    calibration_predictions=neigh.predict(calibration_embeds)
    test_predictions=neigh.predict(test_embeds)

    print("Train accuracy:",np.sum(train_predictions==train_labels)/len(train_labels))
    print("Calibration accuracy:",np.sum(calibration_predictions==calibration_labels)/len(calibration_labels))
    print("Test accuracy:",np.sum(test_predictions==test_labels)/len(test_labels))
    print("Triplet train data silhouette:", silhouette_score(train_embeds,train_labels))
    print("Triplet validation data silhouette:", silhouette_score(calibration_embeds,calibration_labels))
    print("Triplet test data silhouette:", silhouette_score(test_embeds,test_labels))

    confusion_matrix(calibration_labels,calibration_predictions,"kNN Confusion Matrix",(30,30),config.confusion_matrix_plot_path)

    tsne = TSNE()
    train_tsne_embeds = tsne.fit_transform(train_embeds[:5000])
    scatter(train_tsne_embeds, train_labels[:5000], filename=config.scatter_training_plot_path, subtitle="Samples from Triplet Training Data")
    calibration_tsne_embeds = tsne.fit_transform(calibration_embeds)
    scatter(calibration_tsne_embeds, calibration_labels, filename=config.scatter_calibration_plot_path, subtitle="Samples from Triplet Calibration Data")
