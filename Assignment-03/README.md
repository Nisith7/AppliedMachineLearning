# 📩 SMS Spam Classification API  
### Applied Machine Learning | Nisith Ranjan Hazra (MDS202427) 

##  Project Overview

This project implements a production-style **SMS Spam Classification system** with:

- Trained Machine Learning model  
- Probability-based prediction with configurable threshold  
- Unit Testing using `pytest`  
- Flask REST API for model serving  
- Integration testing of API endpoints  

The system classifies SMS messages into:

- **Spam**
- **Ham (Not Spam)**

It also returns a probability score (propensity) between `0` and `1`.

---

##  Project Structure

```
Assignment-03/
│
├── app.py                
├── score.py                
├── predict.py              
├── integration_test.py     
├── test.py                 
│
├── coverage.txt            
├── full_test.log          
├── unit_test.log          
│
└── README.md               
```

---

##  Machine Learning Pipeline

### Feature Engineering
- TF-IDF Vectorization

### Models Evaluated
- Logistic Regression  
- Multinomial Naive Bayes  
- Linear SVM  

### Model Selection Metric
- F1-Score  

The best-performing model is serialized using `joblib` and stored in the `saved_models/` directory.

---

## Scoring Function

Implemented in `score.py`:

```python
def score(text: str, model, threshold: float) -> (bool, float):
```

### Returns:

- `prediction` : `True` (Spam) or `False` (Ham)  
- `propensity` : Probability score between `0` and `1`  

The `threshold` parameter controls classification strictness.

---

##  REST API

### Endpoint

```
POST /score
```

### Request Body

```json
{
  "text": "Congratulations! You won 1000 dollars!",
  "threshold": 0.5
}
```

### Response

```json
{
  "prediction": true,
  "propensity": 0.92
}
```

---

##  Running the Application

###  Install Dependencies

```bash
pip install flask requests pytest scikit-learn joblib numpy
```

###  Start Flask Server

```bash
python app.py
```

Server runs at:

```
http://127.0.0.1:5000
```

---

## Testing

###  Unit Testing

Covers:

- Smoke test  
- Output format validation  
- Prediction type check  
- Probability range validation  
- Threshold edge cases  
- Obvious spam and ham scenarios  

Run:

```bash
pytest test.py
```

---

###  Integration Testing

- Starts Flask server  
- Sends real HTTP request  
- Validates JSON response  
- Proper shutdown handling  

Run:

```bash
pytest test.py
```

---

##  Tech Stack

- Python  
- Scikit-learn  
- Flask  
- Pytest  
- Joblib  
- REST API Design  

---

##  What This Project Demonstrates

- End-to-end ML system implementation  
- Model persistence and loading  
- Threshold-based probability classification  
- Clean REST API design  
- Unit and integration testing for ML systems  
- Production-style project structuring  

--- 

---

##  Author

**Nisith Ranjan Hazra (MDS202427)**  
Master’s in Data Science  
Chennai Mathematical Institute  

---

⭐ If you found this project useful, consider giving it a star!