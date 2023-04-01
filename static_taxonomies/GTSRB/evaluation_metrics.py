import numpy as np
import math
import config

def CE(y_true,y_probs):
    res=0
    for i in range(len(y_true)):
        res+=math.log(y_probs[i,y_true[i]])
    return -1*res

def BS(y_true,y_probs):
    t=np.zeros(y_probs.shape[1],dtype='int')
    res=0
    for i in range(len(y_true)):
        t[y_true[i]]=1
        res+=np.sum((t-y_probs[i])**2)
        t[y_true[i]]=0
    return res/len(y_true)

def ECE_MCE(y_true,y_probs,bins_num):
    bins_edges=np.linspace(0,1,num=bins_num+1)

    hist_confidence=np.zeros(bins_num)
    hist_accuracy=np.zeros(bins_num)
    indexes_hist=[]

    y_preds=np.argmax(y_probs,axis=1)
    y_probs=np.max(y_probs,axis=1)
    correct_predictions=np.array([y_preds==y_true],dtype='int')[0]
    for count,left in enumerate(bins_edges[:-1]):
        indexes_hist.append(np.arange(0,len(y_probs))[np.logical_and(y_probs>=left,y_probs<bins_edges[count+1])])

        if len(indexes_hist[count])>0:
            hist_confidence[count]=np.mean(y_probs[indexes_hist[count]])
            hist_accuracy[count]=np.count_nonzero(correct_predictions[indexes_hist[count]])/len(indexes_hist[count])

    ece=0
    diff=np.abs(hist_confidence-hist_accuracy)
    total_samples=0
    for i in range(len(hist_confidence)):
        total_samples+=len(indexes_hist[i])
        ece+=(len(indexes_hist[i])*diff[i])
    ece/=total_samples
    mce=np.max(np.abs(hist_confidence-hist_accuracy))

    return ece,mce




