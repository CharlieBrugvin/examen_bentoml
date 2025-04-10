import re
import time
import requests

# ----- CONFIGURATION -----

LOGIN_URL = "http://127.0.0.1:3000/login"
PREDICT_URL = "http://127.0.0.1:3000/predict"

# ----- UTILITIES -----

def getting_token(timedelta_before_exp_sec=3600):
    """Get a valid JWT token (identified as alice) with the given expiration time."""
    response = requests.post(
        LOGIN_URL, 
        json={
            "username": "alice", 
            "password": "alicepassword",
            "timedelta_before_exp_sec": timedelta_before_exp_sec
        }
    )

    if response.status_code == 200:
        return response.json().get("token")
    else:
        raise Exception("Failed to get JWT token")
    
input_data_example = {
    "gre_score": 320,
    "toefl_score": 110,
    "university_rating": 5,
    "sop": 5,
    "lor": 5,
    "cgpa": 9,
    "research": 1
}

# ------ LOGIN endpoint ------

def test_login_with_valid_creds():
    """Test the JWT token generation with correct credentials."""
    
    response = requests.post(
        LOGIN_URL,
        headers={"Content-Type": "application/json"},
        json={
            "username": "alice",
            "password": "alicepassword"
        }
    )

    assert response.status_code == 200, "Status code should be 200 with user 'alice'"

    token = response.json()["token"]

    jwt_pattern = r'^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$'

    assert re.fullmatch(jwt_pattern, token), "token should match pattern '^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$'"
    
def test_login_with_invalid_creds():
    """Test the JWT token generation with incorrect credentials."""
    
    response = requests.post(
        LOGIN_URL,
        headers={"Content-Type": "application/json"},
        json={
            "username": "alice",
            "password": "wrong-password"
        }
    )

    assert response.status_code == 401, "Status code should be 401 with user 'alice' and wrong password"


# ------ PREDICT endpoint ------

def test_predict_with_missing_token():
    """Test the prediction endpoint with missing token."""
    
    response = requests.post(
        PREDICT_URL,
        headers={"Content-Type": "application/json"},
        json=input_data_example
    )

    assert response.status_code == 401, "Status code should be 401 without token"


def test_predict_with_expired_token():
    """Test the prediction endpoint with expired token."""
    
    # getting a jwt with 1s lifetime
    token = getting_token(timedelta_before_exp_sec = 1)

    time.sleep(1)

    response = requests.post(
        PREDICT_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        },
        json=input_data_example
    )

    assert response.status_code == 401, "Status code should be 401 with expired token"

def test_predict_with_valid_input():

    # getting valid token
    token = getting_token(timedelta_before_exp_sec=10)

    response = requests.post(
        PREDICT_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        },
        json=input_data_example
    )

    assert response.status_code == 200, "Expected status code 200 for valid input data"

    prediction = response.json().get("prediction")

    assert round(prediction, 1) == 0.8, f"For the given input data, the prediction should be +-0.8 (input data = {input_data_example})"

def test_predict_with_invalid_input():
    """Test the prediction endpoint with invalid input data."""
    
    # getting valid token
    token = getting_token(timedelta_before_exp_sec=10)

    # input with missing 'toefl_score'
    invalid_input_data = {
        'gre_score': 320,
        'university_rating': 5,
        'sop': 5,
        'lor': 5,
        'cgpa': 9,
        'research': 1
    }

    response = requests.post(
        PREDICT_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        },
        json=invalid_input_data
    )

    assert response.status_code == 400, "Expected status code 400 for invalid input data (toefl_score is missing)"