from keras import backend as K
import os
import numpy as np

class Config:
	def __init__(self):
		self.experiment=0
		self.num_classes=131
		self.img_rows,self.img_cols=100,100

		self.k_neighbors_list=[11,11,11,11,11]
		self.k_neighbors=self.k_neighbors_list[self.experiment]

		self.num_categories=(self.k_neighbors//2+1)*self.num_classes

		self.class_mapping_path='GeneratedData/experiment_'+str(self.experiment)+'/class_mapping.pickle'
		self.idxs_class_train_path='GeneratedData/experiment_'+str(self.experiment)+'/idxs_per_class.pickle'

		self.train_dir='~/Dataset/fruits360/Training/'
		self.test_dir='~/Dataset/fruits360/Test/'
		self.train_data_path='GeneratedData/experiment_{}/train_data.pickle'.format(self.experiment)
		self.val_data_path='GeneratedData/experiment_{}/val_data.pickle'.format(self.experiment)
		self.test_data_path='GeneratedData/experiment_{}/test_data.pickle'.format(self.experiment)
		self.data_path='GeneratedData/experiment_{}/data.pickle'.format(self.experiment)

		self.mAP_results_path='mAP/detection-results/'
		self.mAP_gt_path='mAP/ground-truth/'

		self.alpha_list=[1,1]
		self.embeddings_size_list=[128,32,32,128,256]
		self.embeddings_size=self.embeddings_size_list[self.experiment]
		self.alpha=self.alpha_list[self.experiment]

		self.classifier_model_path='Saved_Models/experiment_'+str(self.experiment)+'/classifier_model.h5'
		self.classifier_train_loss_path='Saved_Models/experiment_'+str(self.experiment)+'/classifier_training.log'
		self.siamese_model_path='Saved_Models/experiment_'+str(self.experiment)+'/siamese_model.h5'
		self.siamese_network_weights_path='Saved_Models/experiment_'+str(self.experiment)+'/siamese_model_weights.h5'
		
		self.train_embeddings_path='GeneratedData/experiment_'+str(self.experiment)+'/siamese_embeddings_training.pickle'
		self.calibration_embeddings_path='GeneratedData/experiment_'+str(self.experiment)+'/siamese_embeddings_calibration.pickle'
		self.test_embeddings_path='GeneratedData/experiment_'+str(self.experiment)+'/siamese_embeddings_testing.pickle'
		self.train_probs_path='GeneratedData/experiment_'+str(self.experiment)+'/softmax_probs_training.pickle'
		self.calibration_probs_path='GeneratedData/experiment_'+str(self.experiment)+'/softmax_probs_calibration.pickle'
		self.test_probs_path='GeneratedData/experiment_'+str(self.experiment)+'/softmax_probs_test.pickle'

		self.classifier_embeddings_paths={
			'train':'GeneratedData/experiment_'+str(self.experiment)+'/train_classifier_embeddings.pickle',
			'validation':'GeneratedData/experiment_'+str(self.experiment)+'/validation_classifier_embeddings.pickle',
			'test':'GeneratedData/experiment_'+str(self.experiment)+'/test_classifier_embeddings.pickle'
		}

		self.execution_times_path='GeneratedData/experiment_{}/execution_times.csv'.format(self.experiment)

		self.confusion_matrix_plot_path='Plots/experiment_'+str(self.experiment)+'/siamese/confusion_matrix'
		self.scatter_training_plot_path='Plots/experiment_'+str(self.experiment)+'/siamese/training_scatter'
		self.scatter_calibration_plot_path='Plots/experiment_'+str(self.experiment)+'/siamese/calibration_scatter'

		self.classifier_calibration={
			'data':'Plots/experiment_{}/classifier_calibration_plot_data.csv'.format(self.experiment),
			'plot':'Plots/experiment_{}/classifier_calibration_plot.png'.format(self.experiment)
		}

		self.cumulative_error_plots_data={
		'v1':'Plots/experiment_{}/v1_cumulative_errors_plot_data.csv'.format(self.experiment),
		'v2':'Plots/experiment_{}/v2_cumulative_errors_plot_data.csv'.format(self.experiment),
		'v3':'Plots/experiment_{}/v3_cumulative_errors_plot_data.csv'.format(self.experiment),
		'v4':'Plots/experiment_{}/v4_cumulative_errors_plot_data.csv'.format(self.experiment),
		'knn_v1':'Plots/experiment_{}/knn_v1_cumulative_errors_plot_data.csv'.format(self.experiment),
		'knn_v2':'Plots/experiment_{}/knn_v2_cumulative_errors_plot_data.csv'.format(self.experiment),
		'nc_v1':'Plots/experiment_{}/nc_v1_cumulative_errors_plot_data.csv'.format(self.experiment),
		'nc_v2':'Plots/experiment_{}/nc_v2_cumulative_errors_plot_data.csv'.format(self.experiment)
		}

		self.cumulative_error_plots={
		'v1':'Plots/experiment_{}/v1_cumulative_errors_plot.png'.format(self.experiment),
		'v2':'Plots/experiment_{}/v2_cumulative_errors_plot.png'.format(self.experiment),
		'v3':'Plots/experiment_{}/v3_cumulative_errors_plot.png'.format(self.experiment),
		'v4':'Plots/experiment_{}/v4_cumulative_errors_plot.png'.format(self.experiment),
		'knn_v1':'Plots/experiment_{}/knn_v1_cumulative_errors_plot.png'.format(self.experiment),
		'knn_v2':'Plots/experiment_{}/knn_v2_cumulative_errors_plot.png'.format(self.experiment),
		'nc_v1':'Plots/experiment_{}/nc_v1_cumulative_errors_plot.png'.format(self.experiment),
		'nc_v2':'Plots/experiment_{}/nc_v2_cumulative_errors_plot.png'.format(self.experiment)
		}

		self.cumulative_accuracy_plots={
		'v1':'Plots/experiment_{}/v1_cumulative_accuracy_plot.png'.format(self.experiment),
		'v2':'Plots/experiment_{}/v2_cumulative_accuracy_plot.png'.format(self.experiment),
		'v3':'Plots/experiment_{}/v3_cumulative_accuracy_plot.png'.format(self.experiment),
		'v4':'Plots/experiment_{}/v4_cumulative_accuracy_plot.png'.format(self.experiment),
		'knn_v1':'Plots/experiment_{}/knn_v1_cumulative_accuracy_plot.png'.format(self.experiment),
		'knn_v2':'Plots/experiment_{}/knn_v2_cumulative_accuracy_plot.png'.format(self.experiment),
		'nc_v1':'Plots/experiment_{}/nc_v1_cumulative_accuracy_plot.png'.format(self.experiment),
		'nc_v2':'Plots/experiment_{}/nc_v2_cumulative_accuracy_plot.png'.format(self.experiment)
		}

		self.taxonomies_paths={
		'v1':'GeneratedData/experiment_{}/v1_calibration_results.pickle'.format(self.experiment),
		'v2':'GeneratedData/experiment_{}/v2_calibration_results.pickle'.format(self.experiment),
		'v3':'GeneratedData/experiment_{}/v3_calibration_results.pickle'.format(self.experiment),
		'v4':'GeneratedData/experiment_{}/v4_calibration_results.pickle'.format(self.experiment),
		'knn_v1':'GeneratedData/experiment_{}/knn_v1_calibration_results.pickle'.format(self.experiment),
		'knn_v2':'GeneratedData/experiment_{}/knn_v2_calibration_results.pickle'.format(self.experiment),
		'nc_v1':'GeneratedData/experiment_{}/nc_v1_calibration_results.pickle'.format(self.experiment),
		'nc_v2':'GeneratedData/experiment_{}/nc_v2_calibration_results.pickle'.format(self.experiment)
		}

		self.evaluation_metrics_paths={
		'v1':'GeneratedData/experiment_{}/v1_evaluation.csv'.format(self.experiment),
		'v2':'GeneratedData/experiment_{}/v2_evaluation.csv'.format(self.experiment),
		'v3':'GeneratedData/experiment_{}/v3_evaluation.csv'.format(self.experiment),
		'v4':'GeneratedData/experiment_{}/v4_evaluation.csv'.format(self.experiment),
		'knn_v1':'GeneratedData/experiment_{}/knn_v1_evaluation.csv'.format(self.experiment),
		'knn_v2':'GeneratedData/experiment_{}/knn_v2_evaluation.csv'.format(self.experiment),
		'nc_v1':'GeneratedData/experiment_{}/nc_v1_evaluation.csv'.format(self.experiment),
		'nc_v2':'GeneratedData/experiment_{}/nc_v2_evaluation.csv'.format(self.experiment)
		}

		if not os.path.exists('Saved_Models/experiment_'+str(self.experiment)):
		    os.makedirs('Saved_Models/experiment_'+str(self.experiment))
		if not os.path.exists('GeneratedData/experiment_'+str(self.experiment)):
		    os.makedirs('GeneratedData/experiment_'+str(self.experiment))
		if not os.path.exists('Plots/experiment_'+str(self.experiment)+'/siamese/'):
		    os.makedirs('Plots/experiment_'+str(self.experiment)+'/siamese/')
		if not os.path.exists('Plots/experiment_'+str(self.experiment)+'/calibration/'):
		    os.makedirs('Plots/experiment_'+str(self.experiment)+'/calibration/')


