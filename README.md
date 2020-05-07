# Kaggle_Intel_image_Classifier_CNN</br>
This model classifies the images in 6 classes. The six classes are Buildings, Forest, Glacier, Mountain, Sea and Street.
</br>
# Description of the dataset
</br>
Dataset for the designed CNN is available on Kaggle. Link to the data set is https://www.kaggle.com/puneet6060/intel-image-classification
</br>
Dataset consists of 6 classes and about 2.5k images of each class in training set and 2k images in the test set.</br>
About 7K images are in prediction set.
</br>
# CNN Architecture
</br>
CNN consists of four convolution layers each having max pooling layer and Batch Normalization clubbed with it. 32 filters are used in first and second convolution layers and 64 filters in third and fourth layers. (2,2) max pooling is used in each convolutional layer. Three dense layers are followed by covolution layers.
</br>
# Accuracy
This CNN provides more then 95% accuracy on training set while 92% accuracy on test set. It predicted more than 90% images correctly in the prediction set.
