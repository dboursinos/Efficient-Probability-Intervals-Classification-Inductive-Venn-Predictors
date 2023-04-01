# Dataset
[German Traffic Sign Recognition Dataset (GTSRB)](http://benchmark.ini.rub.de/index.php?section=gtsrb&subsection=about) is an image classification dataset.
The images are photos of traffic signs. The images are classified into 43 classes. The training set contains 39209 labeled images and the test set contains 12630 images. Labels for the test set are not published.

# Evaluation
0. Set the correct paths for inputs and outputs in `config.py`
1. Preprocess the input data and split the into training, calibration and test
   sets by running `preprocess.py`
2. Train the siamese network for distance metric learning by running
   `train_siamese.py`
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
