import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import csv
import pickle
import evaluation_metrics


def nn_v1(y_test,test_probabilities,C):
    taxon='v1'
    with open(C.taxonomies_paths[taxon], "rb") as f:
        category_distributions = pickle.load(f)

    test_predictions=np.argmax(test_probabilities,axis=1)

    # Cumulative Metrics
    Acc=np.empty(len(test_predictions))
    Err=np.empty(len(test_predictions))
    CA=np.empty(len(test_predictions)) #cumulative accuracy
    CE=np.empty(len(test_predictions)) #cumulative errors
    CLAP=np.empty(len(test_predictions)) #cumulative lower accuracy probability
    CUAP=np.empty(len(test_predictions)) #cumulative upper accuracy probability
    LEP=np.empty(len(test_predictions)) #cumulative lower error probability
    UEP=np.empty(len(test_predictions)) #cumulative upper error probability

    test_venn_predictions=np.empty(len(test_predictions),dtype='int')
    test_venn_min=np.empty(len(test_predictions))
    test_venn_max=np.empty(len(test_predictions))
    test_venn_mean_all=np.empty((len(test_predictions),category_distributions.shape[1]))
    for i,category in enumerate(test_predictions):
        samples_in_category=np.sum(category_distributions[category,:])
        temp_distribution_min=category_distributions[category,:]/(samples_in_category+1)
        temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))
        test_venn_mean_all[i]=(temp_distribution_min+temp_distribution_max)/2
        test_venn_predictions[i]=np.argmax(temp_distribution_min)
        test_venn_min[i]=temp_distribution_min[test_venn_predictions[i]]
        test_venn_max[i]=temp_distribution_max[test_venn_predictions[i]]
        Acc[i]=test_venn_predictions[i]==y_test[i]
        Err[i]=test_venn_predictions[i]!=y_test[i]
        CA[i]=np.sum(Acc[:i+1])/(i+1)
        CE[i]=np.sum(Err[:i+1])
        CLAP[i]=np.sum(test_venn_min[:i+1])/(i+1)
        CUAP[i]=np.sum(test_venn_max[:i+1])/(i+1)
        if i==0:
            LEP[i]=1-test_venn_max[i]
            UEP[i]=1-test_venn_min[i]
        else:
            LEP[i]=LEP[i-1]+(1-test_venn_max[i])
            UEP[i]=UEP[i-1]+(1-test_venn_min[i]) 

    print(np.sum(Acc)/len(y_test))

    fig=plt.figure()
    fig.set_size_inches(16,12)
    ax=fig.add_subplot(111)
    ax.set_ylabel('computed probability',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    ax.set_ylim([0.8, 1])
    ax.plot(CA, 'r--', label = "CA")
    ax.plot(CLAP, 'b.', label = "CLAP")
    ax.plot(CUAP, 'k-', label = "CUAP")
    fig.legend()
    fig.savefig(C.cumulative_accuracy_plots[taxon])

    fig=plt.figure()
    fig.set_size_inches(16,12)
    ax=fig.add_subplot(111)
    ax.set_ylabel('Cumulative error',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    # ax.set_ylim([0.8, 1])
    ax.plot(CE, 'r--', label = "CE")
    ax.plot(LEP, 'b-', label = "LEP")
    ax.plot(UEP, 'k-', label = "UEP")
    fig.legend()
    fig.savefig(C.cumulative_error_plots[taxon])   

    downsample_idxs=np.logical_not(np.arange(len(CE))%100)
    error_data={'sample': np.arange(len(CE))[downsample_idxs], 'CE':CE[downsample_idxs], 'LEP':LEP[downsample_idxs], 'UEP':UEP[downsample_idxs]}
    df=pd.DataFrame(error_data,columns=['sample', 'CE', 'LEP', 'UEP'])
    df.to_csv(C.cumulative_error_plots_data[taxon], index=False, header=True) 

    ce=evaluation_metrics.CE(y_test,test_venn_mean_all)
    bs=evaluation_metrics.BS(y_test,test_venn_mean_all)
    ece,mce=evaluation_metrics.ECE_MCE(y_test,test_venn_mean_all,10)
    D=(CUAP[-1]-CLAP[-1])

    with open(C.evaluation_metrics_paths[taxon], 'w', newline='') as csvfile:
        fieldnames = ['Accuracy','CE', 'BS', 'ECE', 'MCE', 'D']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Accuracy':CA[-1], 'CE':ce, 'BS':bs, 'ECE':ece, 'MCE':mce, 'D':D})

def nn_v2(y_test,test_probabilities,C):
    taxon='v2'
    with open(C.taxonomies_paths[taxon], "rb") as f:
        category_distributions = pickle.load(f)

    # place test data in categories
    test_predictions=np.argmax(test_probabilities,axis=1)
    test_prediction_probs=np.max(test_probabilities,axis=1)

    # Cumulative Metrics
    Acc=np.empty(len(test_predictions))
    Err=np.empty(len(test_predictions))
    CA=np.empty(len(test_predictions)) #cumulative accuracy
    CE=np.empty(len(test_predictions)) #cumulative errors
    CLAP=np.empty(len(test_predictions)) #cumulative lower accuracy probability
    CUAP=np.empty(len(test_predictions)) #cumulative upper accuracy probability
    LEP=np.empty(len(test_predictions)) #cumulative lower error probability
    UEP=np.empty(len(test_predictions)) #cumulative upper error probability

    test_venn_predictions=np.empty(len(test_predictions),dtype='int')
    test_venn_min=np.empty(len(test_predictions))
    test_venn_max=np.empty(len(test_predictions))
    test_venn_mean_all=np.empty((len(test_predictions),category_distributions.shape[1]))
    for i in range(len(test_predictions)):
        category=test_predictions[i]*2+(test_prediction_probs[i]>0.75)
        samples_in_category=np.sum(category_distributions[category,:])
        temp_distribution_min=category_distributions[category,:]/(samples_in_category+1)
        temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))
        test_venn_mean_all[i]=(temp_distribution_min+temp_distribution_max)/2
        test_venn_predictions[i]=np.argmax(temp_distribution_min)
        test_venn_min[i]=temp_distribution_min[test_venn_predictions[i]]
        test_venn_max[i]=temp_distribution_max[test_venn_predictions[i]]
        Acc[i]=test_venn_predictions[i]==y_test[i]
        Err[i]=test_venn_predictions[i]!=y_test[i]
        CA[i]=np.sum(Acc[:i+1])/(i+1)
        CE[i]=np.sum(Err[:i+1])
        CLAP[i]=np.sum(test_venn_min[:i+1])/(i+1)
        CUAP[i]=np.sum(test_venn_max[:i+1])/(i+1)
        if i==0:
            LEP[i]=1-test_venn_max[i]
            UEP[i]=1-test_venn_min[i]
        else:
            LEP[i]=LEP[i-1]+(1-test_venn_max[i])
            UEP[i]=UEP[i-1]+(1-test_venn_min[i]) 


    fig=plt.figure()
    fig.set_size_inches(6,4)
    ax=fig.add_subplot(111)
    ax.set_ylabel('Computed probability',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    ax.set_ylim([0.8, 1])
    ax.plot(CA, 'r--', label = "CA")
    ax.plot(CLAP, 'b.', label = "CLAP")
    ax.plot(CUAP, 'k-', label = "CUAP")
    fig.legend()
    fig.tight_layout()
    fig.savefig(C.cumulative_accuracy_plots[taxon])

    fig=plt.figure()
    fig.set_size_inches(6,4)
    ax=fig.add_subplot(111)
    ax.set_ylabel('Cumulative error',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    # ax.set_ylim([0.8, 1])
    ax.plot(CE, 'r--', label = "CE")
    ax.plot(LEP, 'b-', label = "LEP")
    ax.plot(UEP, 'k-', label = "UEP")
    fig.legend()
    fig.tight_layout()
    fig.savefig(C.cumulative_error_plots[taxon])

    downsample_idxs=np.logical_not(np.arange(len(CE))%100)
    error_data={'sample': np.arange(len(CE))[downsample_idxs], 'CE':CE[downsample_idxs], 'LEP':LEP[downsample_idxs], 'UEP':UEP[downsample_idxs]}
    df=pd.DataFrame(error_data,columns=['sample', 'CE', 'LEP', 'UEP'])
    df.to_csv(C.cumulative_error_plots_data[taxon], index=False, header=True) 
    
    ce=evaluation_metrics.CE(y_test,test_venn_mean_all)
    bs=evaluation_metrics.BS(y_test,test_venn_mean_all)
    ece,mce=evaluation_metrics.ECE_MCE(y_test,test_venn_mean_all,10)
    D=(CUAP[-1]-CLAP[-1])

    with open(C.evaluation_metrics_paths[taxon], 'w', newline='') as csvfile:
        fieldnames = ['Accuracy', 'CE', 'BS', 'ECE', 'MCE', 'D']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Accuracy':CA[-1], 'CE':ce, 'BS':bs, 'ECE':ece, 'MCE':mce, 'D':D})

def nn_v3(y_test,test_probabilities,C):
    taxon='v3'
    with open(C.taxonomies_paths[taxon], "rb") as f:
        category_distributions = pickle.load(f)
    # place test data in categories
    test_predictions=np.argmax(test_probabilities,axis=1)
    test_prediction_probs=np.max(test_probabilities,axis=1)
    sorted_probs=np.sort(test_probabilities,axis=1)
    second_highest_probs=sorted_probs[:,-2]


    # Cumulative Metrics
    Acc=np.empty(len(test_predictions))
    Err=np.empty(len(test_predictions))
    CA=np.empty(len(test_predictions)) #cumulative accuracy
    CE=np.empty(len(test_predictions)) #cumulative errors
    CLAP=np.empty(len(test_predictions)) #cumulative lower accuracy probability
    CUAP=np.empty(len(test_predictions)) #cumulative upper accuracy probability
    LEP=np.empty(len(test_predictions)) #cumulative lower error probability
    UEP=np.empty(len(test_predictions)) #cumulative upper error probability

    test_venn_predictions=np.empty(len(test_predictions),dtype='int')
    test_venn_min=np.empty(len(test_predictions))
    test_venn_max=np.empty(len(test_predictions))
    test_venn_mean_all=np.empty((len(test_predictions),category_distributions.shape[1]))
    for i in range(len(test_predictions)):
        category=test_predictions[i]*2+(second_highest_probs[i]>0.25)
        samples_in_category=np.sum(category_distributions[category,:])
        temp_distribution_min=category_distributions[category,:]/(samples_in_category+1)
        temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))
        test_venn_mean_all[i]=(temp_distribution_min+temp_distribution_max)/2
        test_venn_predictions[i]=np.argmax(temp_distribution_min)
        test_venn_min[i]=temp_distribution_min[test_venn_predictions[i]]
        test_venn_max[i]=temp_distribution_max[test_venn_predictions[i]]
        Acc[i]=test_venn_predictions[i]==y_test[i]
        Err[i]=test_venn_predictions[i]!=y_test[i]
        CA[i]=np.sum(Acc[:i+1])/(i+1)
        CE[i]=np.sum(Err[:i+1])
        CLAP[i]=np.sum(test_venn_min[:i+1])/(i+1)
        CUAP[i]=np.sum(test_venn_max[:i+1])/(i+1)
        if i==0:
            LEP[i]=1-test_venn_max[i]
            UEP[i]=1-test_venn_min[i]
        else:
            LEP[i]=LEP[i-1]+(1-test_venn_max[i])
            UEP[i]=UEP[i-1]+(1-test_venn_min[i]) 


    fig=plt.figure()
    fig.set_size_inches(16,12)
    ax=fig.add_subplot(111)
    ax.set_ylabel('computed probability',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    ax.set_ylim([0.8, 1])
    ax.plot(CA, 'r--', label = "CA")
    ax.plot(CLAP, 'b.', label = "CLAP")
    ax.plot(CUAP, 'k-', label = "CUAP")
    fig.legend()
    fig.savefig(C.cumulative_accuracy_plots[taxon])

    fig=plt.figure()
    fig.set_size_inches(6,4)
    ax=fig.add_subplot(111)
    ax.set_ylabel('Cumulative Error',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    # ax.set_ylim([0.8, 1])
    ax.plot(CE, 'r--', label = "CE")
    ax.plot(LEP, 'b-', label = "LEP")
    ax.plot(UEP, 'k-', label = "UEP")
    fig.legend()
    fig.savefig(C.cumulative_error_plots[taxon])

    downsample_idxs=np.logical_not(np.arange(len(CE))%100)
    error_data={'sample': np.arange(len(CE))[downsample_idxs], 'CE':CE[downsample_idxs], 'LEP':LEP[downsample_idxs], 'UEP':UEP[downsample_idxs]}
    df=pd.DataFrame(error_data,columns=['sample', 'CE', 'LEP', 'UEP'])
    df.to_csv(C.cumulative_error_plots_data[taxon], index=False, header=True) 
        
    ce=evaluation_metrics.CE(y_test,test_venn_mean_all)
    bs=evaluation_metrics.BS(y_test,test_venn_mean_all)
    ece,mce=evaluation_metrics.ECE_MCE(y_test,test_venn_mean_all,10)
    D=(CUAP[-1]-CLAP[-1])

    with open(C.evaluation_metrics_paths[taxon], 'w', newline='') as csvfile:
        fieldnames = ['Accuracy', 'CE', 'BS', 'ECE', 'MCE', 'D']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy':CA[-1], 'CE':ce, 'BS':bs, 'ECE':ece, 'MCE':mce, 'D':D})

def nn_v4(y_test,test_probabilities,C):
    taxon='v4'
    with open(C.taxonomies_paths[taxon], "rb") as f:
        category_distributions = pickle.load(f)
    
    # place test data in categories
    test_predictions=np.argmax(test_probabilities,axis=1)
    test_prediction_probs=np.max(test_probabilities,axis=1)
    sorted_probs=np.sort(test_probabilities,axis=1)
    second_highest_probs=sorted_probs[:,-2]


    # Cumulative Metrics
    Acc=np.empty(len(test_predictions))
    Err=np.empty(len(test_predictions))
    CA=np.empty(len(test_predictions)) #cumulative accuracy
    CE=np.empty(len(test_predictions)) #cumulative errors
    CLAP=np.empty(len(test_predictions)) #cumulative lower accuracy probability
    CUAP=np.empty(len(test_predictions)) #cumulative upper accuracy probability
    LEP=np.empty(len(test_predictions)) #cumulative lower error probability
    UEP=np.empty(len(test_predictions)) #cumulative upper error probability

    test_venn_predictions=np.empty(len(test_predictions),dtype='int')
    test_venn_min=np.empty(len(test_predictions))
    test_venn_max=np.empty(len(test_predictions))
    test_venn_mean_all=np.empty((len(test_predictions),category_distributions.shape[1]))
    for i in range(len(test_predictions)):
        category=test_predictions[i]*2+(test_prediction_probs[i]-second_highest_probs[i]>0.5)
        samples_in_category=np.sum(category_distributions[category,:])
        temp_distribution_min=category_distributions[category,:]/(samples_in_category+1)
        temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))
        test_venn_mean_all[i]=(temp_distribution_min+temp_distribution_max)/2
        test_venn_predictions[i]=np.argmax(temp_distribution_min)
        test_venn_min[i]=temp_distribution_min[test_venn_predictions[i]]
        test_venn_max[i]=temp_distribution_max[test_venn_predictions[i]]
        Acc[i]=test_venn_predictions[i]==y_test[i]
        Err[i]=test_venn_predictions[i]!=y_test[i]
        CA[i]=np.sum(Acc[:i+1])/(i+1)
        CE[i]=np.sum(Err[:i+1])
        CLAP[i]=np.sum(test_venn_min[:i+1])/(i+1)
        CUAP[i]=np.sum(test_venn_max[:i+1])/(i+1)
        if i==0:
            LEP[i]=1-test_venn_max[i]
            UEP[i]=1-test_venn_min[i]
        else:
            LEP[i]=LEP[i-1]+(1-test_venn_max[i])
            UEP[i]=UEP[i-1]+(1-test_venn_min[i]) 


    fig=plt.figure()
    fig.set_size_inches(16,12)
    ax=fig.add_subplot(111)
    ax.set_ylabel('computed probability',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    ax.set_ylim([0.8, 1])
    ax.plot(CA, 'r--', label = "CA")
    ax.plot(CLAP, 'b.', label = "CLAP")
    ax.plot(CUAP, 'k-', label = "CUAP")
    fig.legend()
    fig.savefig(C.cumulative_accuracy_plots[taxon])

    fig=plt.figure()
    fig.set_size_inches(16,12)
    ax=fig.add_subplot(111)
    ax.set_ylabel('Cumulative Error',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    # ax.set_ylim([0.8, 1])
    ax.plot(CE, 'r--', label = "CE")
    ax.plot(LEP, 'b-', label = "LEP")
    ax.plot(UEP, 'k-', label = "UEP")
    fig.legend()
    fig.savefig(C.cumulative_error_plots[taxon])

    downsample_idxs=np.logical_not(np.arange(len(CE))%100)
    error_data={'sample': np.arange(len(CE))[downsample_idxs], 'CE':CE[downsample_idxs], 'LEP':LEP[downsample_idxs], 'UEP':UEP[downsample_idxs]}
    df=pd.DataFrame(error_data,columns=['sample', 'CE', 'LEP', 'UEP'])
    df.to_csv(C.cumulative_error_plots_data[taxon], index=False, header=True) 
        
    ce=evaluation_metrics.CE(y_test,test_venn_mean_all)
    bs=evaluation_metrics.BS(y_test,test_venn_mean_all)
    ece,mce=evaluation_metrics.ECE_MCE(y_test,test_venn_mean_all,10)
    D=(CUAP[-1]-CLAP[-1])

    with open(C.evaluation_metrics_paths[taxon], 'w', newline='') as csvfile:
        fieldnames = ['Accuracy', 'CE', 'BS', 'ECE', 'MCE', 'D']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy':CA[-1], 'CE':ce, 'BS':bs, 'ECE':ece, 'MCE':mce, 'D':D})

def knn_v1(test_embeds,y_test,C):
    taxon='knn_v1'
    with open(C.taxonomies_paths[taxon], "rb") as f:
        neigh,category_distributions = pickle.load(f)


    # Cumulative Metrics
    Acc=np.empty(len(test_embeds))
    Err=np.empty(len(test_embeds))
    CA=np.empty(len(test_embeds)) #cumulative accuracy
    CE=np.empty(len(test_embeds)) #cumulative errors
    CLAP=np.empty(len(test_embeds)) #cumulative lower accuracy probability
    CUAP=np.empty(len(test_embeds)) #cumulative upper accuracy probability
    LEP=np.empty(len(test_embeds)) #cumulative lower error probability
    UEP=np.empty(len(test_embeds)) #cumulative upper error probability

    test_venn_predictions=np.empty(len(test_embeds),dtype='int')
    test_venn_min=np.empty(len(test_embeds))
    test_venn_max=np.empty(len(test_embeds))
    test_venn_mean_all=np.empty((len(test_embeds),category_distributions.shape[1]))
    for i,test_embedding in enumerate(test_embeds):
        category=neigh.predict(test_embedding.reshape(1, -1))
        samples_in_category=np.sum(category_distributions[category,:])
        temp_distribution_min=category_distributions[category,:][0]/(samples_in_category+1)
        temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))
        test_venn_mean_all[i]=(temp_distribution_min+temp_distribution_max)/2
        test_venn_predictions[i]=np.argmax(temp_distribution_min)
        test_venn_min[i]=temp_distribution_min[test_venn_predictions[i]]
        test_venn_max[i]=temp_distribution_max[test_venn_predictions[i]]
        Acc[i]=test_venn_predictions[i]==y_test[i]
        Err[i]=test_venn_predictions[i]!=y_test[i]
        CA[i]=np.sum(Acc[:i+1])/(i+1)
        CE[i]=np.sum(Err[:i+1])
        CLAP[i]=np.sum(test_venn_min[:i+1])/(i+1)
        CUAP[i]=np.sum(test_venn_max[:i+1])/(i+1)
        if i==0:
            LEP[i]=1-test_venn_max[i]
            UEP[i]=1-test_venn_min[i]
        else:
            LEP[i]=LEP[i-1]+(1-test_venn_max[i])
            UEP[i]=UEP[i-1]+(1-test_venn_min[i]) 

    print(np.sum(Acc)/len(y_test))

    fig=plt.figure()
    fig.set_size_inches(6,4)
    ax=fig.add_subplot(111)
    ax.set_ylabel('computed probability',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    ax.set_ylim([0.8, 1])
    ax.plot(CA, 'r--', label = "CA")
    ax.plot(CLAP, 'b.', label = "CLAP")
    ax.plot(CUAP, 'k-', label = "CUAP")
    fig.legend()
    fig.tight_layout()
    fig.savefig(C.cumulative_accuracy_plots[taxon])
        
    fig=plt.figure()
    fig.set_size_inches(6,4)
    ax=fig.add_subplot(111)
    ax.set_ylabel('Cumulative error',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    # ax.set_ylim([0.8, 1])
    ax.plot(CE, 'r--', label = "CE")
    ax.plot(LEP, 'b-', label = "LEP")
    ax.plot(UEP, 'k-', label = "UEP")
    fig.legend()
    fig.tight_layout()
    fig.savefig(C.cumulative_error_plots[taxon])

    downsample_idxs=np.logical_not(np.arange(len(CE))%100)
    error_data={'sample': np.arange(len(CE))[downsample_idxs], 'CE':CE[downsample_idxs], 'LEP':LEP[downsample_idxs], 'UEP':UEP[downsample_idxs]}
    df=pd.DataFrame(error_data,columns=['sample', 'CE', 'LEP', 'UEP'])
    df.to_csv(C.cumulative_error_plots_data[taxon], index=False, header=True)

    ce=evaluation_metrics.CE(y_test,test_venn_mean_all)
    bs=evaluation_metrics.BS(y_test,test_venn_mean_all)
    ece,mce=evaluation_metrics.ECE_MCE(y_test,test_venn_mean_all,10)
    D=(CUAP[-1]-CLAP[-1])

    with open(C.evaluation_metrics_paths[taxon], 'w', newline='') as csvfile:
        fieldnames = ['Accuracy', 'CE', 'BS', 'ECE', 'MCE', 'D']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy':CA[-1], 'CE':ce, 'BS':bs, 'ECE':ece, 'MCE':mce, 'D':D})


def knn_v2(test_embeds,y_test,y_train,C):
    taxon='knn_v2'
    with open(C.taxonomies_paths[taxon], "rb") as f:
        neigh,category_distributions = pickle.load(f)


    # Cumulative Metrics
    Acc=np.empty(len(test_embeds))
    Err=np.empty(len(test_embeds))
    CA=np.empty(len(test_embeds)) #cumulative accuracy
    CE=np.empty(len(test_embeds)) #cumulative errors
    CLAP=np.empty(len(test_embeds)) #cumulative lower accuracy probability
    CUAP=np.empty(len(test_embeds)) #cumulative upper accuracy probability
    LEP=np.empty(len(test_embeds)) #cumulative lower error probability
    UEP=np.empty(len(test_embeds)) #cumulative upper error probability

    test_venn_predictions=np.empty(len(test_embeds),dtype='int')
    test_venn_min=np.empty(len(test_embeds))
    test_venn_max=np.empty(len(test_embeds))
    test_venn_mean_all=np.empty((len(test_embeds),category_distributions.shape[1]))
    for i,test_embedding in enumerate(test_embeds):
        test_prediction=neigh.predict(test_embedding.reshape(1, -1))
        test_neighbor_indexes=neigh.kneighbors(test_embedding.reshape(1, -1),return_distance=False)
        test_neighbor_labels=y_train[test_neighbor_indexes]
        category=test_prediction*(C.k_neighbors//2+1)+np.count_nonzero(test_neighbor_labels[0]!=test_prediction[0])
        samples_in_category=np.sum(category_distributions[category,:])
        temp_distribution_min=category_distributions[category,:][0]/(samples_in_category+1)
        temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))
        test_venn_mean_all[i]=(temp_distribution_min+temp_distribution_max)/2
        test_venn_predictions[i]=np.argmax(temp_distribution_min)
        test_venn_min[i]=temp_distribution_min[test_venn_predictions[i]]
        test_venn_max[i]=temp_distribution_max[test_venn_predictions[i]]
        Acc[i]=test_venn_predictions[i]==y_test[i]
        Err[i]=test_venn_predictions[i]!=y_test[i]
        CA[i]=np.sum(Acc[:i+1])/(i+1)
        CE[i]=np.sum(Err[:i+1])
        CLAP[i]=np.sum(test_venn_min[:i+1])/(i+1)
        CUAP[i]=np.sum(test_venn_max[:i+1])/(i+1)
        if i==0:
            LEP[i]=1-test_venn_max[i]
            UEP[i]=1-test_venn_min[i]
        else:
            LEP[i]=LEP[i-1]+(1-test_venn_max[i])
            UEP[i]=UEP[i-1]+(1-test_venn_min[i]) 

    print(np.sum(Acc)/len(y_test))

    fig=plt.figure()
    fig.set_size_inches(16,12)
    ax=fig.add_subplot(111)
    ax.set_ylabel('computed probability',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    ax.set_ylim([0.8, 1])
    ax.plot(CA, 'r--', label = "CA")
    ax.plot(CLAP, 'b.', label = "CLAP")
    ax.plot(CUAP, 'k-', label = "CUAP")
    fig.legend()
    fig.savefig(C.cumulative_accuracy_plots[taxon])
        
    fig=plt.figure()
    fig.set_size_inches(16,12)
    ax=fig.add_subplot(111)
    ax.set_ylabel('Cumulative Error',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    # ax.set_ylim([0.8, 1])
    ax.plot(CE, 'r--', label = "CE")
    ax.plot(LEP, 'b-', label = "LEP")
    ax.plot(UEP, 'k-', label = "UEP")
    fig.legend()
    fig.savefig(C.cumulative_error_plots[taxon])

    downsample_idxs=np.logical_not(np.arange(len(CE))%100)
    error_data={'sample': np.arange(len(CE))[downsample_idxs], 'CE':CE[downsample_idxs], 'LEP':LEP[downsample_idxs], 'UEP':UEP[downsample_idxs]}
    df=pd.DataFrame(error_data,columns=['sample', 'CE', 'LEP', 'UEP'])
    df.to_csv(C.cumulative_error_plots_data[taxon], index=False, header=True)

    ce=evaluation_metrics.CE(y_test,test_venn_mean_all)
    bs=evaluation_metrics.BS(y_test,test_venn_mean_all)
    ece,mce=evaluation_metrics.ECE_MCE(y_test,test_venn_mean_all,10)
    D=(CUAP[-1]-CLAP[-1])

    with open(C.evaluation_metrics_paths[taxon], 'w', newline='') as csvfile:
        fieldnames = ['Accuracy', 'CE', 'BS', 'ECE', 'MCE', 'D']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy':CA[-1], 'CE':ce, 'BS':bs, 'ECE':ece, 'MCE':mce, 'D':D})

def nc_v1(test_embeds,y_test,C):
    taxon='nc_v1'
    with open(C.taxonomies_paths[taxon], "rb") as f:
        centroids,category_distributions = pickle.load(f)

    # Cumulative Metrics
    Acc=np.empty(len(test_embeds))
    Err=np.empty(len(test_embeds))
    CA=np.empty(len(test_embeds)) #cumulative accuracy
    CE=np.empty(len(test_embeds)) #cumulative errors
    CLAP=np.empty(len(test_embeds)) #cumulative lower accuracy probability
    CUAP=np.empty(len(test_embeds)) #cumulative upper accuracy probability
    LEP=np.empty(len(test_embeds)) #cumulative lower error probability
    UEP=np.empty(len(test_embeds)) #cumulative upper error probability

    test_venn_predictions=np.empty(len(test_embeds),dtype='int')
    test_venn_min=np.empty(len(test_embeds))
    test_venn_max=np.empty(len(test_embeds))
    test_venn_mean_all=np.empty((len(test_embeds),C.num_classes))
    temp_distances=np.zeros(C.num_classes)
    for i,test_embedding in enumerate(test_embeds):
        for j in range(C.num_classes):
            temp_distances[j]=np.linalg.norm(test_embedding-centroids[j])
        category=np.argmin(temp_distances)
        samples_in_category=np.sum(category_distributions[category,:])
        temp_distribution_min=category_distributions[category,:]/(samples_in_category+1)
        temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))
        test_venn_mean_all[i]=(temp_distribution_min+temp_distribution_max)/2
        test_venn_predictions[i]=np.argmax(test_venn_mean_all[i])
        test_venn_min[i]=temp_distribution_min[test_venn_predictions[i]]
        test_venn_max[i]=temp_distribution_max[test_venn_predictions[i]]
        Acc[i]=test_venn_predictions[i]==y_test[i]
        Err[i]=test_venn_predictions[i]!=y_test[i]
        CA[i]=np.sum(Acc[:i+1])/(i+1)
        CE[i]=np.sum(Err[:i+1])
        CLAP[i]=np.sum(test_venn_min[:i+1])/(i+1)
        CUAP[i]=np.sum(test_venn_max[:i+1])/(i+1)
        if i==0:
            LEP[i]=1-test_venn_max[i]
            UEP[i]=1-test_venn_min[i]
        else:
            LEP[i]=LEP[i-1]+(1-test_venn_max[i])
            UEP[i]=UEP[i-1]+(1-test_venn_min[i]) 

    print(np.sum(Acc)/len(y_test))

    fig=plt.figure()
    fig.set_size_inches(16,12)
    ax=fig.add_subplot(111)
    ax.set_ylabel('computed probability',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    ax.set_ylim([0.8, 1])
    ax.plot(CA, 'r--', label = "CA")
    ax.plot(CLAP, 'b.', label = "CLAP")
    ax.plot(CUAP, 'k-', label = "CUAP")
    fig.legend()
    fig.savefig(C.cumulative_accuracy_plots[taxon])
        
    fig=plt.figure()
    fig.set_size_inches(16,12)
    ax=fig.add_subplot(111)
    ax.set_ylabel('Cumulative Error',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    # ax.set_ylim([0.8, 1])
    ax.plot(CE, 'r--', label = "CE")
    ax.plot(LEP, 'b-', label = "LEP")
    ax.plot(UEP, 'k-', label = "UEP")
    fig.legend()
    fig.savefig(C.cumulative_error_plots[taxon])

    downsample_idxs=np.logical_not(np.arange(len(CE))%100)
    error_data={'sample': np.arange(len(CE))[downsample_idxs], 'CE':CE[downsample_idxs], 'LEP':LEP[downsample_idxs], 'UEP':UEP[downsample_idxs]}
    df=pd.DataFrame(error_data,columns=['sample', 'CE', 'LEP', 'UEP'])
    df.to_csv(C.cumulative_error_plots_data[taxon], index=False, header=True)

    ce=evaluation_metrics.CE(y_test,test_venn_mean_all)
    bs=evaluation_metrics.BS(y_test,test_venn_mean_all)
    ece,mce=evaluation_metrics.ECE_MCE(y_test,test_venn_mean_all,10)
    D=(CUAP[-1]-CLAP[-1])

    with open(C.evaluation_metrics_paths[taxon], 'w', newline='') as csvfile:
        fieldnames = ['Accuracy', 'CE', 'BS', 'ECE', 'MCE', 'D']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy':CA[-1], 'CE':ce, 'BS':bs, 'ECE':ece, 'MCE':mce, 'D':D})

def nc_v2(test_embeds,y_test,C):
    taxon='nc_v2'
    with open(C.taxonomies_paths[taxon], "rb") as f:
        centroids,category_distributions = pickle.load(f)

    # Cumulative Metrics
    Acc=np.empty(len(test_embeds))
    Err=np.empty(len(test_embeds))
    CA=np.empty(len(test_embeds)) #cumulative accuracy
    CE=np.empty(len(test_embeds)) #cumulative errors
    CLAP=np.empty(len(test_embeds)) #cumulative lower accuracy probability
    CUAP=np.empty(len(test_embeds)) #cumulative upper accuracy probability
    LEP=np.empty(len(test_embeds)) #cumulative lower error probability
    UEP=np.empty(len(test_embeds)) #cumulative upper error probability

    test_venn_predictions=np.empty(len(test_embeds),dtype='int')
    test_venn_min=np.empty(len(test_embeds))
    test_venn_max=np.empty(len(test_embeds))
    test_venn_mean_all=np.empty((len(test_embeds),C.num_classes))
    temp_distances=np.zeros(C.num_classes)
    for i,test_embedding in enumerate(test_embeds):
        for j in range(C.num_classes):
            temp_distances[j]=np.linalg.norm(test_embedding-centroids[j])
        category=2*np.argmin(temp_distances)+int(np.min(temp_distances)>0.08)
        samples_in_category=np.sum(category_distributions[category,:])
        temp_distribution_min=category_distributions[category,:]/(samples_in_category+1)
        temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))
        test_venn_mean_all[i]=(temp_distribution_min+temp_distribution_max)/2
        test_venn_predictions[i]=np.argmax(test_venn_mean_all[i])
        test_venn_min[i]=temp_distribution_min[test_venn_predictions[i]]
        test_venn_max[i]=temp_distribution_max[test_venn_predictions[i]]
        Acc[i]=test_venn_predictions[i]==y_test[i]
        Err[i]=test_venn_predictions[i]!=y_test[i]
        CA[i]=np.sum(Acc[:i+1])/(i+1)
        CE[i]=np.sum(Err[:i+1])
        CLAP[i]=np.sum(test_venn_min[:i+1])/(i+1)
        CUAP[i]=np.sum(test_venn_max[:i+1])/(i+1)
        if i==0:
            LEP[i]=1-test_venn_max[i]
            UEP[i]=1-test_venn_min[i]
        else:
            LEP[i]=LEP[i-1]+(1-test_venn_max[i])
            UEP[i]=UEP[i-1]+(1-test_venn_min[i]) 

    print(np.sum(Acc)/len(y_test))

    fig=plt.figure()
    fig.set_size_inches(6,4)
    ax=fig.add_subplot(111)
    ax.set_ylabel('computed probability',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    ax.set_ylim([0.8, 1])
    ax.plot(CA, 'r--', label = "CA")
    ax.plot(CLAP, 'b.', label = "CLAP")
    ax.plot(CUAP, 'k-', label = "CUAP")
    fig.legend()
    fig.savefig(C.cumulative_accuracy_plots[taxon])
        
    fig=plt.figure()
    fig.set_size_inches(6,4)
    ax=fig.add_subplot(111)
    ax.set_ylabel('Cumulative Error',color='k',fontsize=16)
    ax.set_xlabel('example #',color='k',fontsize=16)
    # ax.set_ylim([0.8, 1])
    ax.plot(CE, 'r--', label = "CE")
    ax.plot(LEP, 'b-', label = "LEP")
    ax.plot(UEP, 'k-', label = "UEP")
    fig.legend()
    fig.savefig(C.cumulative_error_plots[taxon])

    downsample_idxs=np.logical_not(np.arange(len(CE))%100)
    error_data={'sample': np.arange(len(CE))[downsample_idxs], 'CE':CE[downsample_idxs], 'LEP':LEP[downsample_idxs], 'UEP':UEP[downsample_idxs]}
    df=pd.DataFrame(error_data,columns=['sample', 'CE', 'LEP', 'UEP'])
    df.to_csv(C.cumulative_error_plots_data[taxon], index=False, header=True)

    ce=evaluation_metrics.CE(y_test,test_venn_mean_all)
    bs=evaluation_metrics.BS(y_test,test_venn_mean_all)
    ece,mce=evaluation_metrics.ECE_MCE(y_test,test_venn_mean_all,10)
    D=(CUAP[-1]-CLAP[-1])

    with open(C.evaluation_metrics_paths[taxon], 'w', newline='') as csvfile:
        fieldnames = ['Accuracy', 'CE', 'BS', 'ECE', 'MCE', 'D']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Accuracy':CA[-1], 'CE':ce, 'BS':bs, 'ECE':ece, 'MCE':mce, 'D':D})

