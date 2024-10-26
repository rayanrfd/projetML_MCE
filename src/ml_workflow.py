import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# One function for pre-processing


# One function to prepare the dataset for training
def prepare_dataset(data):
    y = np.array(data[data.columns[-1]])
    X = np.array(data.iloc[:, 0:len(data) - 1])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    return X_train, X_val, y_train, y_val

# One function for training (typically applies up to 5 different methods for binary classification)
def training_validating_model(X, y, classifier, step):
    if step == 'training':
        classifier.fit(X, y)
    if step == 'validation':
        classifier.predict(X, y)

# One function to display all the results in a convenient form for comparison
