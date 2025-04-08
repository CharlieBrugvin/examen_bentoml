"""We prepare the admission dataset for training and testing."""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

# --- Parameters ---

RAW_DATA_FILE = "data/raw/admission.csv"
TEST_SIZE = 0.2
SEED = 42
PROCESSED_DATA_FOLDER = "data/processed"

# --- Dataset loading ---

admission_df = pd.read_csv(RAW_DATA_FILE)
logging.info("Dataset loaded successfully from %s", RAW_DATA_FILE)

# Getting feature matrix and target vector

X = admission_df[[
    'GRE Score',
    'TOEFL Score',
    'University Rating',
    'SOP',
    'LOR',
    'CGPA',
    'Research',
]].values

y = admission_df[
    'Chance of Admit'
].values

# ---- Dataset splitting ---

logging.info("Splitting intro train / test datasets. Test size: %s", TEST_SIZE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

# --- Logging & saving ---

# Logging the shapes of the datasets
logging.info(
    "Shapes - X_train: %s, X_test: %s, y_train: %s, y_test: %s",
    X_train.shape, X_test.shape, y_train.shape, y_test.shape
)

# Saving the datasets
np.save(f"{PROCESSED_DATA_FOLDER}/X_train.npy", X_train)
np.save(f"{PROCESSED_DATA_FOLDER}/X_test.npy", X_test)
np.save(f"{PROCESSED_DATA_FOLDER}/y_train.npy", y_train)
np.save(f"{PROCESSED_DATA_FOLDER}/y_test.npy", y_test)

logging.info("X_train, X_test, y_train, and y_test saved successfully at %s", PROCESSED_DATA_FOLDER)