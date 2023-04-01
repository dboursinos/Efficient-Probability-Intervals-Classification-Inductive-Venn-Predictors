import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import csv
import pickle
import time
import evaluation_metrics
import config

sys.setrecursionlimit(40000)
C = config.Config()

epsilon=0.5

def extract_results(CA,CLAP,CUAP,CE,LEP,UEP,type,taxon):
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
    path='Plots/experiment_{}/{}/{}_cumulative_accuracy_plot.png'.format(C.experiment,type,taxon)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.savefig(path)
        
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
    path='Plots/experiment_{}/{}/{}_cumulative_errors_plot.png'.format(C.experiment,type,taxon)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.savefig(path)

    downsample_idxs=np.logical_not(np.arange(len(CE))%100)
    error_data={'sample': np.arange(len(CE))[downsample_idxs], 'CE':CE[downsample_idxs], 'LEP':LEP[downsample_idxs], 'UEP':UEP[downsample_idxs]}
    df=pd.DataFrame(error_data,columns=['sample', 'CE', 'LEP', 'UEP'])
    path='Plots/experiment_{}/{}/{}_cumulative_errors_plot_data.csv'.format(C.experiment,type,taxon)
    df.to_csv(path, index=False, header=True)


def compute_single_pvalues(test_embeds,y_train):
    p_values=np.empty((len(test_embeds),C.num_classes))
    with open(C.calibration_nc_scores['knn'], "rb") as f:
        calibration_nc,neigh=pickle.load(f)
    for i in range(len(test_embeds)):
        indices = neigh.kneighbors(test_embeds[i].reshape(1, -1), return_distance=False)
        test_ICP_candidates_temp=y_train[indices[0]]
        for j in range(C.num_classes):
            temp_nc=C.knn_neighbors-np.count_nonzero(test_ICP_candidates_temp==j)
            if C.p_value_type=="regular":
                p_values[i,j]=np.count_nonzero(calibration_nc>=temp_nc)/len(calibration_nc)
            if C.p_value_type=="smooth":
                p_values[i,j]=(np.count_nonzero(calibration_nc>temp_nc)+np.random.uniform(size=1)*np.count_nonzero(calibration_nc==temp_nc))/len(calibration_nc)
    return p_values


def knn_v1(test_embeds,data,C):
    taxon='knn_v1'
    for type in ['static','dynamic_1','dynamic_2']:
        with open(C.taxonomies_paths[taxon], "rb") as f:
            neigh,category_distributions = pickle.load(f)

        p_values=compute_single_pvalues(test_embeds,data['y_train'])

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
        start=time.time()
        for i,test_embedding in enumerate(test_embeds):
            category=neigh.predict(test_embedding.reshape(1, -1))
            samples_in_category=np.sum(category_distributions[category,:])
            temp_distribution_min=category_distributions[category,:][0]/(samples_in_category+1)
            temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))
            test_venn_mean_all[i]=(temp_distribution_min+temp_distribution_max)/2
            test_venn_predictions[i]=np.argmax(temp_distribution_min)
            test_venn_min[i]=temp_distribution_min[test_venn_predictions[i]]
            test_venn_max[i]=temp_distribution_max[test_venn_predictions[i]]
            Acc[i]=test_venn_predictions[i]==data['y_test'][i]
            Err[i]=test_venn_predictions[i]!=data['y_test'][i]
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

            'Try to append the categories'
            if type=='dynamic_1':
                if np.count_nonzero(p_values[i]>=epsilon)>=1:
                    p_values_sort=np.sort(p_values[i])
                    pmax=p_values_sort[-1]
                    psecmax=p_values_sort[-2]
                    if pmax-3*psecmax>=0:
                        category_distributions[category,np.argmax(p_values[i])]+=1

            if type=='dynamic_2':
                for cl in range(C.num_classes):
                    if p_values[i,cl]>=epsilon:
                        category_distributions[category,cl]+=1
                    
        end=time.time()
        print("knn_v1 {} {}".format(type,(end-start)/len(test_embeds)))
        print(np.sum(Acc)/len(data['y_test']))
        extract_results(CA,CLAP,CUAP,CE,LEP,UEP,type,taxon)

        ce=evaluation_metrics.CE(data['y_test'],test_venn_mean_all)
        bs=evaluation_metrics.BS(data['y_test'],test_venn_mean_all)
        ece,mce=evaluation_metrics.ECE_MCE(data['y_test'],test_venn_mean_all,10)
        D=(CUAP[-1]-CLAP[-1])

        path='GeneratedData/experiment_{}/{}/{}_evaluation.csv'.format(C.experiment,type,taxon)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w', newline='') as csvfile:
            fieldnames = ['Accuracy', 'CE', 'BS', 'ECE', 'MCE', 'D']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'Accuracy':CA[-1], 'CE':ce, 'BS':bs, 'ECE':ece, 'MCE':mce, 'D':D})


        

def knn_v2(test_embeds,data,C):
    taxon='knn_v2'
    for type in ['static','dynamic_1','dynamic_2']:
        with open(C.taxonomies_paths[taxon], "rb") as f:
            neigh,category_distributions = pickle.load(f)

        p_values=compute_single_pvalues(test_embeds,data['y_train'])

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
        start=time.time()
        for i,test_embedding in enumerate(test_embeds):
            test_prediction=neigh.predict(test_embedding.reshape(1, -1))
            test_neighbor_indexes=neigh.kneighbors(test_embedding.reshape(1, -1),return_distance=False)
            test_neighbor_labels=data['y_train'][test_neighbor_indexes]
            category=test_prediction*(C.k_neighbors//2+1)+np.count_nonzero(test_neighbor_labels[0]!=test_prediction[0])
            samples_in_category=np.sum(category_distributions[category,:])
            temp_distribution_min=category_distributions[category,:][0]/(samples_in_category+1)
            temp_distribution_max=temp_distribution_min+(1/(samples_in_category+1))
            test_venn_mean_all[i]=(temp_distribution_min+temp_distribution_max)/2
            test_venn_predictions[i]=np.argmax(temp_distribution_min)
            test_venn_min[i]=temp_distribution_min[test_venn_predictions[i]]
            test_venn_max[i]=temp_distribution_max[test_venn_predictions[i]]
            Acc[i]=test_venn_predictions[i]==data['y_test'][i]
            Err[i]=test_venn_predictions[i]!=data['y_test'][i]
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
            
            'Try to append the categories'
            if type=='dynamic_1':
                if np.count_nonzero(p_values[i]>=epsilon)>=1:
                    p_values_sort=np.sort(p_values[i])
                    pmax=p_values_sort[-1]
                    psecmax=p_values_sort[-2]
                    if pmax-3*psecmax>=0:
                        category_distributions[category,np.argmax(p_values[i])]+=1

            if type=='dynamic_2':
                for cl in range(C.num_classes):
                    if p_values[i,cl]>=epsilon:
                        category_distributions[category,cl]+=1

        end=time.time()
        print("knn_v2 {} {}".format(type,(end-start)/len(test_embeds)))
        print(np.sum(Acc)/len(data['y_test']))
        extract_results(CA,CLAP,CUAP,CE,LEP,UEP,type,taxon)

        ce=evaluation_metrics.CE(data['y_test'],test_venn_mean_all)
        bs=evaluation_metrics.BS(data['y_test'],test_venn_mean_all)
        ece,mce=evaluation_metrics.ECE_MCE(data['y_test'],test_venn_mean_all,10)
        D=(CUAP[-1]-CLAP[-1])

        path='GeneratedData/experiment_{}/{}/{}_evaluation.csv'.format(C.experiment,type,taxon)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w', newline='') as csvfile:
            fieldnames = ['Accuracy', 'CE', 'BS', 'ECE', 'MCE', 'D']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'Accuracy':CA[-1], 'CE':ce, 'BS':bs, 'ECE':ece, 'MCE':mce, 'D':D})

def nc_v1(test_embeds,data,C):
    taxon='nc_v1'
    for type in ['static','dynamic_1','dynamic_2']:
        with open(C.taxonomies_paths[taxon], "rb") as f:
            centroids,category_distributions = pickle.load(f)

        p_values=compute_single_pvalues(test_embeds,data['y_train'])

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
        start=time.time()
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
            Acc[i]=test_venn_predictions[i]==data['y_test'][i]
            Err[i]=test_venn_predictions[i]!=data['y_test'][i]
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

            'Try to append the categories'
            if type=='dynamic_1':
                if np.count_nonzero(p_values[i]>=epsilon)>=1:
                    p_values_sort=np.sort(p_values[i])
                    pmax=p_values_sort[-1]
                    psecmax=p_values_sort[-2]
                    if pmax-3*psecmax>=0:
                        category_distributions[category,np.argmax(p_values[i])]+=1

            if type=='dynamic_2':
                for cl in range(C.num_classes):
                    if p_values[i,cl]>=epsilon:
                        category_distributions[category,cl]+=1

        end=time.time()
        print("nc_v1 {} {}".format(type,(end-start)/len(test_embeds)))
        print(np.sum(Acc)/len(data['y_test']))
        extract_results(CA,CLAP,CUAP,CE,LEP,UEP,type,taxon)

        ce=evaluation_metrics.CE(data['y_test'],test_venn_mean_all)
        bs=evaluation_metrics.BS(data['y_test'],test_venn_mean_all)
        ece,mce=evaluation_metrics.ECE_MCE(data['y_test'],test_venn_mean_all,10)
        D=(CUAP[-1]-CLAP[-1])

        path='GeneratedData/experiment_{}/{}/{}_evaluation.csv'.format(C.experiment,type,taxon)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w', newline='') as csvfile:
            fieldnames = ['Accuracy', 'CE', 'BS', 'ECE', 'MCE', 'D']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'Accuracy':CA[-1], 'CE':ce, 'BS':bs, 'ECE':ece, 'MCE':mce, 'D':D})

def nc_v2(test_embeds,data,C):
    taxon='nc_v2'
    for type in ['static','dynamic_1','dynamic_2']:
        with open(C.taxonomies_paths[taxon], "rb") as f:
            centroids,category_distributions = pickle.load(f)

        p_values=compute_single_pvalues(test_embeds,data['y_train'])

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
        start=time.time()
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
            Acc[i]=test_venn_predictions[i]==data['y_test'][i]
            Err[i]=test_venn_predictions[i]!=data['y_test'][i]
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

            'Try to append the categories'
            if type=='dynamic_1':
                if np.count_nonzero(p_values[i]>=epsilon)>=1:
                    p_values_sort=np.sort(p_values[i])
                    pmax=p_values_sort[-1]
                    psecmax=p_values_sort[-2]
                    if pmax-3*psecmax>=0:
                        category_distributions[category,np.argmax(p_values[i])]+=1

            if type=='dynamic_2':
                for cl in range(C.num_classes):
                    if p_values[i,cl]>=epsilon:
                        category_distributions[category,cl]+=1

        end=time.time()
        print("nc_v2 {} {}".format(type,(end-start)/len(test_embeds)))
        print(np.sum(Acc)/len(data['y_test']))
        extract_results(CA,CLAP,CUAP,CE,LEP,UEP,type,taxon)

        ce=evaluation_metrics.CE(data['y_test'],test_venn_mean_all)
        bs=evaluation_metrics.BS(data['y_test'],test_venn_mean_all)
        ece,mce=evaluation_metrics.ECE_MCE(data['y_test'],test_venn_mean_all,10)
        D=(CUAP[-1]-CLAP[-1])

        path='GeneratedData/experiment_{}/{}/{}_evaluation.csv'.format(C.experiment,type,taxon)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w', newline='') as csvfile:
            fieldnames = ['Accuracy', 'CE', 'BS', 'ECE', 'MCE', 'D']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'Accuracy':CA[-1], 'CE':ce, 'BS':bs, 'ECE':ece, 'MCE':mce, 'D':D})
