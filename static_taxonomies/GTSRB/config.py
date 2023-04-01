from keras import backend as K
import os
import numpy as np


class Config:

	def __init__(self):
		self.experiment=0
		self.num_classes = 43
		self.input_side_size=96
		self.embeddings_size_list=[128,8,512]
		self.k_neighbors_list=[101]
		self.k_neighbors=self.k_neighbors_list[self.experiment]
		self.num_categories=(self.k_neighbors//2+1)*self.num_classes
		self.alpha=0.5
		self.cnn_blocks_list=[4,4,4]
		self.nc_function_list=['nearest_centroid','nearest_centroid','1nn'] #knn or nearest_centroid
		self.knn_neighbors_list=[40,40,40]
		self.p_value_type_list=["regular","regular","regular"] #smooth or regular
		self.significance_levels=np.arange(start=0.001,stop=0.25001,step=0.001)

		self.embeddings_size=self.embeddings_size_list[self.experiment]
		self.cnn_blocks=self.cnn_blocks_list[self.experiment]
		self.nc_function=self.nc_function_list[self.experiment]
		self.knn_neighbors=self.knn_neighbors_list[self.experiment]
		self.p_value_type=self.p_value_type_list[self.experiment]

		self.preprocessed_data="~/Dataset/GTSRB/preprocessed_dataset.pickle"
		self.classifier_model_path='Saved_Models/experiment_'+str(self.experiment)+'/classifier_model.h5'
		self.classifier_train_loss_path='Saved_Models/experiment_'+str(self.experiment)+'/classifier_training.log'
		self.siamese_model_path='Saved_Models/experiment_'+str(self.experiment)+'/siamese_model.h5'
		self.idxs_class_train_path='GeneratedData/experiment_'+str(self.experiment)+'/idxs_per_class.pickle'
		self.train_embeddings_path='GeneratedData/experiment_'+str(self.experiment)+'/train_embeddings.pickle'
		self.calibration_embeddings_path='GeneratedData/experiment_'+str(self.experiment)+'/calibration_embeddings.pickle'
		self.test_embeddings_path='GeneratedData/experiment_'+str(self.experiment)+'/test_embeddings.pickle'
		self.train_probs_path='GeneratedData/experiment_'+str(self.experiment)+'/softmax_probs_training.pickle'
		self.calibration_probs_path='GeneratedData/experiment_'+str(self.experiment)+'/softmax_probs_calibration.pickle'
		self.test_probs_path='GeneratedData/experiment_'+str(self.experiment)+'/softmax_probs_test.pickle'

		self.execution_times_path='GeneratedData/experiment_{}/execution_times.csv'.format(self.experiment)

		self.classifier_embeddings_paths={
			'train':'GeneratedData/experiment_'+str(self.experiment)+'/train_classifier_embeddings.pickle',
			'validation':'GeneratedData/experiment_'+str(self.experiment)+'/validation_classifier_embeddings.pickle',
			'test':'GeneratedData/experiment_'+str(self.experiment)+'/test_classifier_embeddings.pickle'
		}

		self.classifier_calibration={
			'data':'Plots/experiment_{}/classifier_calibration_plot_data.csv'.format(self.experiment),
			'plot':'Plots/experiment_{}/classifier_calibration_plot.png'.format(self.experiment)
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

		self.nn2_nc2_calibration_results_path='GeneratedData/experiment_'+str(self.experiment)+'/nn2_nc2_calibration_results.pickle'

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

		self.nn2_nc2_evaluation_metrics_path='GeneratedData/experiment_'+str(self.experiment)+'/nn2_nc2_evaluation.csv'

		# Plots
		# Class sizes
		self.class_sizes_plot_path="Plots/class_sizes.png"
		# Scatter plots for embedding evaluation
		self.training_scatter_plot_path='Plots/experiment_{}/training_scatter.png'.format(self.experiment)
		self.calibration_scatter_plot_path='Plots/experiment_{}/calibration_scatter.png'.format(self.experiment)
		# Confusion Matrix
		self.training_confusion_plot_path='Plots/experiment_{}/training_confusion_matrix.png'.format(self.experiment)
		self.calibration_confusion_plot_path='Plots/experiment_{}/calibration_confusion_matrix.png'.format(self.experiment)

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

		if not os.path.exists('Saved_Models/experiment_'+str(self.experiment)):
		    os.makedirs('Saved_Models/experiment_'+str(self.experiment))
		if not os.path.exists('GeneratedData/experiment_'+str(self.experiment)):
		    os.makedirs('GeneratedData/experiment_'+str(self.experiment))
		if not os.path.exists('Plots/experiment_'+str(self.experiment)):
		    os.makedirs('Plots/experiment_'+str(self.experiment))
