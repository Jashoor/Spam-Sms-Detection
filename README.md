# Spam-Sms-Detection
Overview
This project focuses on building a machine learning model to detect spam SMS messages. The goal is to develop a robust classification system that accurately identifies whether a given SMS is spam or not. The project uses various machine learning algorithms, including Multinomial Naive Bayes, Gaussian Naive Bayes, Support Vector Machine (SVM), Logistic Regression, Random Forest Classifier, and AdaBoost Classifier.

Table of Contents
Introduction
Dependencies
Dataset
Data Preprocessing
Feature Extraction
Model Training
Evaluation
Usage
Results
Contributing
License
Introduction
Spam SMS is a prevalent issue, and detecting such messages is crucial to enhancing user experience and security. This project employs machine learning techniques to automatically identify and filter out spam SMS messages.

Dependencies
Ensure you have the following dependencies installed:

pandas
numpy
matplotlib
seaborn
scikit-learn
You can install these dependencies using the following:

bash
Copy code
pip install pandas numpy matplotlib seaborn scikit-learn
Dataset
The dataset used for this project contains labeled SMS messages, where each message is classified as either spam or not spam (ham). The dataset is loaded using the pandas library.

Data Preprocessing
Before training the machine learning models, the dataset undergoes preprocessing steps such as handling missing values, removing duplicates, and balancing the class distribution.

Feature Extraction
Text data is converted into numerical features using the CountVectorizer from scikit-learn. This step is essential for training machine learning models.

Model Training
The project employs various classifiers, including Multinomial Naive Bayes, Gaussian Naive Bayes, Support Vector Machine, Logistic Regression, Random Forest Classifier, and AdaBoost Classifier. The models are trained on the preprocessed data.

Evaluation
The models' performance is evaluated using accuracy as the primary metric. Additional metrics, such as precision, recall, and F1 score, may also be considered.

Usage
To use the trained models for spam SMS detection, follow these steps:

Load the trained model.
Preprocess the new SMS data.
Use the model to predict whether each SMS is spam or not.
Results
The results section provides insights into the performance of each model. Consider including visualizations such as confusion matrices or ROC curves for a comprehensive understanding.
