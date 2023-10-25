
# Heart Disease Prediction

## Introduction

This Jupyter Notebook is focused on predicting heart disease using a dataset from the University of California, Irvine (UCI) Machine Learning Repository. It provides an overview of the data, exploratory data analysis (EDA), data preprocessing, and the application of various machine learning models to predict heart disease.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Overview](#data-overview)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Data Preprocessing](#data-preprocessing)
5. [Machine Learning Models](#machine-learning-models)
    5.1. [Logistic Regression](#logistic-regression)
    5.2. [K-Nearest Neighbors](#k-nearest-neighbors)
    5.3. [Random Forest Classifier](#random-forest-classifier)
    5.4. [Decision Tree](#decision-tree)
    5.5. [Support Vector Machine (SVM)](#support-vector-machine-svm)
6. [Conclusion](#conclusion)

## Data Overview

### Data Source

The dataset used for this analysis is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease). The dataset contains information about patients and whether or not they have heart disease.

### Features

The dataset includes various features, both categorical and numeric, such as age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol level (chol), fasting blood sugar (fbs), resting electrocardiographic results (restecg), maximum heart rate achieved (thalach), exercise-induced angina (exang), ST depression induced by exercise relative to rest (oldpeak), the slope of the peak exercise ST segment (slope), number of major vessels (0-3) colored by fluoroscopy (ca), and the thalassemia type (thal).

### Target

The target variable is "num," which represents the presence of heart disease. In the dataset, values 0, 1, 2, 3, and 4 correspond to different stages of heart disease. This variable is used to train machine learning models to predict heart disease.

## Exploratory Data Analysis

In this section, we perform exploratory data analysis (EDA) to gain insights into the dataset.

### Age Distribution

We visualize the distribution of ages in the dataset and observe differences between males and females. The box plot and histogram provide a clear understanding of the age distribution.

### Male and Female Proportion

We investigate the gender distribution in the dataset to understand the proportion of males and females. A pie chart is used to display the gender ratio.

### Dataset Contributors

As the dataset is an amalgamation of four independent studies, we examine the contribution of each study to the overall dataset. A pie chart shows the proportion of data from each contributor.

### Chest Pain Type (CP) Proportions

We explore the proportions of different chest pain types (cp) in the dataset. A pie chart illustrates the distribution of chest pain types.

### Resting Blood Pressure vs. Gender

We examine the distribution of resting blood pressure (trestbps) for all patients and differentiate by gender (male and female) to identify any potential differences.

### Heart Disease Frequency

We visualize the frequency of heart disease cases with respect to gender. This bar plot provides insights into the distribution of heart disease cases among males and females.

### Heart Disease Frequency per Chest Pain Type

We analyze the frequency of heart disease cases according to different chest pain types (cp). This bar plot helps in understanding the relationship between chest pain and heart disease.

## Data Preprocessing

In the data preprocessing phase, we perform the following tasks:

- Handling missing data: We fill missing values in numerical columns with the median and create binary columns to indicate missing data.
- Encoding categorical variables: We convert categorical variables into numerical form using label encoding.

## Machine Learning Models

In this section, we apply several machine learning models to predict heart disease based on the preprocessed dataset.

### Logistic Regression

We use logistic regression to build a classification model and evaluate its performance.

### K-Nearest Neighbors (KNN)

K-Nearest Neighbors is employed to create a classification model for heart disease prediction.

### Random Forest Classifier

A random forest classifier is utilized to construct an ensemble model for heart disease prediction.

### Decision Tree

We apply a decision tree classifier to build a classification model and evaluate its performance.

### Support Vector Machine (SVM)

We utilize a support vector machine classifier to create a classification model for predicting heart disease.

## Conclusion

This notebook explored the prediction of heart disease using machine learning models. We conducted an exploratory data analysis to gain insights into the dataset, preprocessed the data, and applied various machine learning models. The performance of each model was evaluated, and hyperparameter tuning was performed. The results provide valuable information for predicting heart disease in patients.

Please refer to the Jupyter Notebook for detailed code and visualizations.

### Author
[Your Name]

### References
- Dataset Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- Icon Source: [Pexels](https://www.pexels.com/photo/people-woman-girl-man-357678/)

