"""We train a simple linear regression model using scikit-learn and save it with BentoML."""

import logging

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import bentoml

logging.basicConfig(level=logging.INFO)

# ---- Parameters ----

PROCESSED_DATA_FOLDER = "data/processed"
MODEL_NAME = "admission_regression_model"


# ---- Data Loading ----

X_train = np.load(f"{PROCESSED_DATA_FOLDER}/X_train.npy")
y_train = np.load(f"{PROCESSED_DATA_FOLDER}/y_train.npy")
X_test = np.load(f"{PROCESSED_DATA_FOLDER}/X_test.npy")
y_test = np.load(f"{PROCESSED_DATA_FOLDER}/y_test.npy")

logging.info("Data loaded successfully from %s", PROCESSED_DATA_FOLDER)
logging.info("Shapes - X_train: %s, X_test: %s, y_train: %s, y_test: %s",
            X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# ----- Modeling -----

linear_regressor = LinearRegression()

linear_regressor.fit(X_train, y_train)

logging.info("Linear regression model trained successfully")

# ---- Evaluation ----

y_test_pred = linear_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

logging.info("Model evaluation: MSE: %s, R2: %s", mse, r2)

# ---- Saving the model ----

bentoml.sklearn.save_model(
    MODEL_NAME,
    linear_regressor,
)
logging.info("Successfully saved to Model Store as '%s'", MODEL_NAME)