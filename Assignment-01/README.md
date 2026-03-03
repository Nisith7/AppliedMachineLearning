# Assignment 1: SMS Spam Classification

**Course:** Applied Machine Learning  
**Institute:** Chennai Mathematical Institute  
**Student:** Nisith Ranjan Hazra (MDS202427)

---

##  Problem Overview

The objective of this project is to build a machine learning–based SMS spam detection system that classifies messages as:

- **Ham** – Legitimate message  
- **Spam** – Unsolicited message  

The task presents two major challenges:

1. **Class Imbalance** – Only ~13% of messages are spam  
2. **Asymmetric Error Costs** – False positives (blocking legitimate messages) are 10× more costly than false negatives  

The goal is to build a model that effectively detects spam while minimizing incorrect blocking of legitimate messages.

---

##  Dataset

**SMS Spam Collection Dataset**

- Source: UCI Machine Learning Repository  
- Total samples: 5,572 messages  
- Class distribution:
  - Ham: ~87%
  - Spam: ~13%

---

##  Methodology

###  Data Preparation (`prepare.ipynb`)

- Download and load dataset  
- Perform Exploratory Data Analysis (EDA):
  - Class distribution analysis  
  - Message length distribution  
  - Word frequency patterns  
- Create stratified splits:
  - Training: 70%
  - Validation: 15%
  - Test: 15%

---

### 2️ Model Training (`train.ipynb`)

#### Feature Engineering
- TF-IDF vectorization  
- Maximum features: 5000  

#### Models Evaluated
- Rule-based keyword baseline  
- Logistic Regression  
- Multinomial Naive Bayes  
- Linear SVM  

#### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- AUPRC (Area Under Precision-Recall Curve)  
- AUROC  

---

##  Validation Results

### Heuristic Rule-Based Model
Training: Accuracy 0.9536 | Precision 0.9014 | Recall 0.7341 | F1 0.8092  
Validation: Accuracy 0.9533 | Precision 0.8871 | Recall 0.7432 | F1 0.8088  

Confusion Matrix:
```
[[476   7]
 [ 19  55]]
```

---

### Logistic Regression
Validation: Accuracy 0.9677 | Precision 1.0000 | Recall 0.7568 | F1 0.8615  

Confusion Matrix:
```
[[483   0]
 [ 18  56]]
```

---

### Multinomial Naive Bayes
Validation: Accuracy 0.9695 | Precision 1.0000 | Recall 0.7703 | F1 0.8702  

Confusion Matrix:
```
[[483   0]
 [ 17  57]]
```

---

##  Hyperparameter Optimization

**Logistic Regression**
```
{'C': 100, 'class_weight': 'balanced', 'penalty': 'l2', 'solver': 'liblinear'}
```
Best CV F1: 0.9408  

**Multinomial Naive Bayes**
```
{'alpha': 0.1, 'fit_prior': True}
```
Best CV F1: 0.9423  

**Linear SVM**
```
{'C': 10, 'class_weight': None, 'loss': 'squared_hinge'}
```
Best CV F1: 0.9399  

---

##  Final Test Set Performance

| Model | Accuracy | Precision | Recall | F1 | AUPRC | FP | FN |
|--------|----------|-----------|--------|------|--------|----|----|
| Logistic Regression | 98.21% | 94.50% | 91.96% | 0.9321 | 0.9638 | 6 | 9 |
| **Multinomial Naive Bayes** | **98.33%** | **96.23%** | 91.07% | **0.9358** | **0.9708** | **4** | 10 |
| Linear SVM | 98.21% | 94.50% | 91.96% | 0.9321 | 0.9673 | 6 | 9 |

---

##  Cost-Sensitive Analysis

Using a 10:1 penalty ratio (False Positive = 10 × False Negative):

| Model | Weighted Cost (10×FP + FN) |
|--------|----------------------------|
| Logistic Regression | 69 |
| **Multinomial Naive Bayes** | **50** |
| Linear SVM | 69 |

---

##  Final Model Selection

**Selected Model: Multinomial Naive Bayes**

Reasons:

- Highest F1-score (0.9358)
- Highest AUPRC (0.9708)
- Highest precision (96.23%)
- Lowest false positives (4)
- Lowest cost under asymmetric penalty setting

Although Logistic Regression and SVM show competitive recall, their higher false-positive rate makes them less suitable for deployment where user trust is critical.

---

##  Project Structure

```
Assignment-01/
│
├── README.md
├── prepare.ipynb
├── train.ipynb
│
└── dataset/
    ├── SMSSpamCollection
    ├── train_data.csv
    ├── validation_data.csv
    └── test_data.csv
```

---

## How to Run

1. Open and run `prepare.ipynb` to preprocess data and generate splits  
2. Run `train.ipynb` to train models and evaluate performance  

---

##  Requirements

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  

---

##  Key Takeaways

- Handling class imbalance requires appropriate evaluation metrics (AUPRC, F1).  
- Cost-sensitive evaluation is critical in spam filtering.  
- Multinomial Naive Bayes provides the best trade-off between precision and recall for this dataset.  

---
