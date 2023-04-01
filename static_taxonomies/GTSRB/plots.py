import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scikitplot as skplt
import matplotlib.patheffects as PathEffects

def scatter(x, labels, filename, subtitle=None):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", np.max(labels)+1))

    # We create a scatter plot.
    fig = plt.figure(figsize=(12, 12))
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