
## **Course: Applied Machine Learning** 
## **Student: Nisith Ranjan Hazra (MDS202427)**  
## **SMS Spam Classification**  


---

## Overview

This repository contains two assignments focused on building a reproducible and production-oriented machine learning workflow for SMS Spam Classification.  

The project covers:

- End-to-end ML prototype development  
- Data Version Control (DVC)  
- Experiment Tracking and Model Versioning using MLflow  

Dataset used for Assignment-01:  
UCI SMS Spam Collection Dataset  
https://archive.ics.uci.edu/ml/datasets/sms+spam+collection  


---

# Assignment-01: ML Prototype Development

## Problem Statement

Build a complete prototype for SMS Spam Classification that:

1. Loads the SMS spam dataset from a given file path  
2. Preprocesses the text data (if necessary)  
3. Splits the dataset into train, validation, and test sets  
4. Saves the splits as:
   - train.csv  
   - validation.csv  
   - test.csv  

Then implement a training pipeline that:

1. Fits a model on the training data  
2. Scores the model on given data  
3. Evaluates predictions using appropriate metrics  
4. Validates the model  
5. Scores on both training and validation sets  
6. Fine-tunes hyperparameters (if required)  
7. Trains three benchmark models  
8. Evaluates all three models on the test set  
9. Selects the best model based on a suitable metric (e.g., AUCPR)  

The goal is to design a structured, modular, and reproducible ML workflow following good experiment design practices.


---

# Assignment-02: Experiment Tracking and Version Control

## Problem Statement

Extend Assignment-01 by introducing proper data and model version control.

## Data Version Control (DVC)

In prepare.ipynb:

1. Track raw_data.csv using DVC  
2. Track train.csv, validation.csv, and test.csv  
3. Change the random seed and regenerate new splits  
4. Track the updated dataset versions  
5. Checkout the first version using DVC and print the distribution of the target variable (number of 0s and 1s) in:
   - train.csv  
   - validation.csv  
   - test.csv  
6. Checkout the updated version and again print the target distribution  
7. (Bonus) Configure Google Drive as remote storage to decouple compute and storage  

The goal is to demonstrate reproducible data pipelines with proper versioning.


## Model Versioning and Experiment Tracking (MLflow)

In train.ipynb:

1. Track experiments using MLflow  
2. Log:
   - Parameters  
   - Metrics  
   - Models  
3. Train and register three benchmark models  
4. Log evaluation metric AUCPR for each model  
5. Compare models using MLflow  
6. Select the best model based on AUCPR  

The goal is to demonstrate structured experiment tracking, model comparison, and model version management following industry best practices.
