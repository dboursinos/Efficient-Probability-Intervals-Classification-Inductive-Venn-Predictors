# Dataset
[Fruit360](https://www.kaggle.com/moltean/fruits) is a dataset with 90380 images of 131 fruits and vegetables. Images are 100 pixel by 100 pixel and are RGB (color) images (3 values for each pixel).

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
