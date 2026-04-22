# Face Classification using Naive Bayes

- This project is a simple implementation of a face classification model using the Olivetti Faces dataset. The goal was to understand how Naive Bayes can be applied to image data and to evaluate its performance on a real dataset.

# Overview
* Used the Olivetti Faces dataset (40 people, 10 images each)
* Implemented both Gaussian Naive Bayes and Multinomial Naive Bayes
* Split the dataset into training and testing sets
* Evaluated the model using accuracy and confusion matrix
* Visualized some misclassified images

# Tech Stack
- Python
- NumPy
- Matplotlib
- Scikit-learn

# Dataset Details
Total images: 400
Image size: 64 × 64 pixels
Each image is grayscale
40 distinct individuals

# How it works
* Load the dataset using fetch_olivetti_faces()
* Flatten images into feature vectors
* Split data into training (70%) and testing (30%)
* Train the Naive Bayes model
* Predict labels for test data
* Evaluate accuracy and analyze errors

# Results
- Gaussian Naive Bayes Accuracy: ~85%
- Multinomial Naive Bayes Accuracy: slightly lower compared to Gaussian

- Gaussian NB performed better because the dataset contains continuous pixel values.

# Observations
- Naive Bayes is fast and simple but assumes feature independence
- Some faces are misclassified due to similar features across individuals
- Performance could be improved using dimensionality reduction (like PCA)

# Sample Output
- Display of faces from dataset
- Confusion matrix
- Misclassified face images

# How to Run
* pip install numpy matplotlib scikit-learn
* python your_file_name.py
