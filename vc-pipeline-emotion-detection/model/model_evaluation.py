from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import numpy as np
import pandas as pd
import pickle
import json
import os

# fetching the model
clf = pickle.load(open(os.path.join("models", 'model.pkl'), 'rb'))


# Fetch the data
test_data = pd.read_csv("./data/processed/test_processed.csv")

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values


# Make predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)


metrics_dict = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "auc": auc

}

with open('metrics.json', 'w') as file:
    json.dump(metrics_dict, file, indent=4)

