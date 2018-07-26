# NN_pattern_recognition
Image Recognition and Multi-class classification using Multi-variate regression and Neural Networks 

Author: Abe (Abhijith Rajeev)
Project dates: Dec 2016 - Jan 2017
Libraries and Code references: Introduction to Machine Learning - Andrew Ng 

************ Project Overview ******************

- Cifar dataset from University of Toronto is used, but the whole dataset is compressed/reduced for the computational purposes. 
- Neural network is designed according to the requirement, input_layer_size = 3072; means 32x32 color image, RGB 1024 each. 10 various objects as in the original database, hidden layer size of 25.
- Forward propagation is performed using the randomly initialized parameters. 
- Backward propagation is performed to learn the optimal parameters.
- Optimal parameters are used with the multivariate regression function - fmincg for recognition. 
- Images are classified according to the learned pattern.
