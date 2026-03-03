import pytest
import joblib
from score import score
from app import app


#### Test Cases

SCORE_CASES = [

    #### Spam
    {"id": "spam_1", "text": "Win money now!!!", "expected": True},
    {"id": "spam_2", "text": "Congratulations! You have won a free lottery ticket.", "expected": True},
    {"id": "spam_3", "text": "Claim your 5000 dollar prize by calling now.", "expected": True},
    {"id": "spam_4", "text": "URGENT! Your number has been selected for a cash reward.", "expected": True},
    {"id": "spam_5", "text": "Exclusive offer!!! Click the link to get free vouchers.", "expected": True},
    {"id": "spam_6", "text": "You are a lucky winner. Text WIN to 80808 now.", "expected": True},
    {"id": "spam_7", "text": "Limited time deal, buy now and get 70% discount.", "expected": True},
    {"id": "spam_8", "text": "Free entry in weekly competition. Reply YES to join.", "expected": True},

    #### Ham
    {"id": "ham_1", "text": "Are we meeting tomorrow?", "expected": False},
    {"id": "ham_2", "text": "Please send me the assignment file.", "expected": False},
    {"id": "ham_3", "text": "Can you call me when you reach home?", "expected": False},
    {"id": "ham_4", "text": "The class has been postponed to Monday.", "expected": False},
    {"id": "ham_5", "text": "I will be late for dinner today.", "expected": False},
    {"id": "ham_6", "text": "Let us go for badminton this evening.", "expected": False},
    {"id": "ham_7", "text": "Did you complete the project report?", "expected": False},
    {"id": "ham_8", "text": "Happy birthday! Have a great year ahead.", "expected": False},
]


# Unit Test Setup (score.py)

@pytest.fixture(scope="module")
def model():
    """
    Load trained model once for all unit tests.
    """
    return joblib.load("saved_models/best_spam_classifier_model.pkl")


#### Unit Tests for score()

@pytest.mark.parametrize("case", SCORE_CASES, ids=[c["id"] for c in SCORE_CASES])
def test_score_prediction(model, case):

    pred, prop = score(case["text"], model, 0.5)

    assert isinstance(pred, bool), f"{case['id']} prediction is not bool"
    assert isinstance(prop, float), f"{case['id']} propensity is not float"
    assert 0 <= prop <= 1, f"{case['id']} propensity out of range"
    assert pred == case["expected"], \
        f"{case['id']} expected {case['expected']}, got {pred}"


@pytest.mark.parametrize("case", SCORE_CASES, ids=[c["id"] for c in SCORE_CASES])
def test_threshold_edges(model, case):
    """
    threshold=0 should predict True
    threshold=1 should predict False
    """
    pred_zero, _ = score(case["text"], model, 0.0)
    assert pred_zero is True

    pred_one, _ = score(case["text"], model, 1.0)
    assert pred_one is False


####Integration Test Setup
@pytest.fixture
def client():
    """
    Flask test client (NO subprocess).
    This fixes coverage issue.
    """
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


#### Integration Tests (Flask endpoint)

@pytest.mark.integration
@pytest.mark.parametrize("case", SCORE_CASES, ids=[c["id"] for c in SCORE_CASES])
def test_flask_score(client, model, case):

    payload = {
        "text": case["text"],
        "threshold": 0.5
    }

    response = client.post("/score", json=payload)

    assert response.status_code == 200, \
        f"{case['id']} returned status {response.status_code}"

    result = response.get_json()

    assert "prediction" in result
    assert "propensity" in result

    expected_pred, expected_prop = score(case["text"], model, 0.5)

    assert result["prediction"] == expected_pred
    assert abs(result["propensity"] - expected_prop) < 1e-6


@pytest.mark.integration
def test_flask_invalid_input(client):
    """
    Test missing text field.
    """
    response = client.post("/score", json={"threshold": 0.5})
    assert response.status_code in [400, 422]