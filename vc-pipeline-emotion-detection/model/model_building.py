import numpy as np
import pandas as pd
import os
import yaml
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os

n_estimators = yaml.safe_load(open('params.yaml', 'r'))['model_building']['n_estimators']
learning_rate = yaml.safe_load(open('params.yaml', 'r'))['model_building']['learning_rate']
# Fetch the data
train_data = pd.read_csv("./data/processed/train_processed.csv")

X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values


clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

clf.fit(X_train, y_train)
os.makedirs("models", exist_ok=True)


# save model, wb is to write in binary
pickle.dump(clf, open(os.path.join("models", "model.pkl"), 'wb'))