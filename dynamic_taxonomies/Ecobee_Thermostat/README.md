# Dataset
[Link](https://data.mendeley.com/datasets/hpbszmrns7)
The original data comes from the work of Meidan et al. [1]. It was preprocessed in this setting for comparative analysis of anomaly detection. The following steps have been taken as preprocessing: (1) five devices have been selected: Danmini doorbell, Ecobee thermostat, Philips baby monitor, Provision security camera, Samsung webcam, (2) for each botnet, the malicious traffic of all five behaviour types have been merged, (3) for each device and botnet combination, malicious requests have been sampled to comprise 5% of the final dataset.

[1] Meidan, Y., Bohadana, M., Mathov, Y., Mirsky, Y., Shabtai, A., Breitenbacher, D., & Elovici, Y. (2018). N-baiot—network-based detection of iot botnet attacks using deep autoencoders. IEEE Pervasive Computing, 17(3), 12-22.

# Evaluation
0. Set the correct paths for inputs and outputs in `config.py`.
1. Preprocess the input data and split the into training, calibration and test
   sets by running `preprocess.py`.
2. Train the siamese network for distance metric learning by running
   `train_siamese.py`.
3. Train the classifier DNN for the baseline by running `train_classifier.py`
4. Generate the embeding representation using the siamese network by running
   `generate_embeddings.py`.
5. Generate the embedding representations using the baseline DNN by running
   `generate_classifier_embeddings.py`.
6. Generate the softmax probabilities that are used to place input data in Venn
   taxonomies for the baseline by running `generate_classifier_probabilities.py`.
7. Evaluate the quality of the embedding representations and softmax
   probabilities by running
   `evaluate_embeddings.py`, `evaluate_embeddings_classifier.py` and
   `evaluate_probabilities.py`.
8. Place the calibration data into the categories by running
   `calibration.py`.
9. Compute the accuracy and calibration results on the test data by running
   `test.py`.
10. Compute the execution times by running `evaluate_times.py`.
