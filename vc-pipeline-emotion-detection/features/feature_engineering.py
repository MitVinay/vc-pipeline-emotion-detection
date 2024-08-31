import numpy as np
import pandas as pd
import os
import yaml

from sklearn.feature_extraction.text import CountVectorizer

max_features = yaml.safe_load(open('params.yaml', 'r'))['feature_engineering']['max_features']

# Fetch the data
train_data = pd.read_csv("./data/interim/train_interim.csv")
test_data = pd.read_csv("./data/interim/test_interim.csv")

train_data.fillna(" ", inplace=True)
test_data.fillna(" ", inplace=True)

# Preparing data for bag of words
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=max_features)

# Fit the vectorizer on the training data and transform it
X_train_bow = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer
X_test_bow = vectorizer.transform(X_test)


# Combining the data label and the input
train_df = pd.DataFrame(X_train_bow.toarray())

train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())

test_df['label'] = y_test


# Defining the path
data_path = os.path.join("data", "processed")

# making directory
os.makedirs(data_path)

# write the file
train_df.to_csv(os.path.join(data_path, "train_processed.csv"))
test_df.to_csv(os.path.join(data_path, "test_processed.csv"))


