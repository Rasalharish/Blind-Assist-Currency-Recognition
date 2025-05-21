# model/test_model.py

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import os

model = load_model("modeles.h5")

# Replace with actual test data loading code
# X_test, y_test = load_test_data()
# Example:
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
# print(f"Test Accuracy: {accuracy:.2f}")
