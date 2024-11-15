# Importing the basic libraries
import numpy as np
import pandas as pd

# Importing preprocessers
from sklearn.model_selection import train_test_split

# Importing the classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier


# One function for pre-processing

## Kidney dataset '?' handling
def question_mark_handling(data):
    for col in data.columns:
        if data[col].dtype() == 'string':
              for m in data[col]:
                    if m[0:1]=='\t':
                        m = m[2:]
    return data
    
    data = data.replace(['?', '\t', '\t?'], np.nan)
    return data

## Divides the dataset between the features and the label
def extract_label(data, column):
    label = data[column]
    data = data.drop(column, axis=1)
    return data, label


def data_preprocessing_impute(data): #mean_col, freq_col, uknw_col, value):
    
    mean_col = data.select_dtypes(include=['int64', 'float64']).columns.to_list()
    if mean_col is not None:
        data[mean_col] = data[mean_col].fillna(data[mean_col].mean())

    freq_col = data.select_dtypes(include=['object']).columns.to_list()
    if freq_col is not None:
        data[freq_col] = data[freq_col].fillna(data[freq_col].mode())

    return data


def data_preprocessing_encode(data): #, num_col, cat_col, ord_col):
     
    num_col = data.select_dtypes(include=['int64', 'float64']).columns.to_list()
    if num_col is not None:
        data[num_col] = (data[num_col] - data[num_col].min()) / (data[num_col].max() - data[num_col].min())

    cat_col = data.select_dtypes(include=['object']).columns.to_list()
    if cat_col is not None: 
        data = pd.get_dummies(data, columns=cat_col, dtype=int)

    return data

def data_preprocessing(data, label_column): #, num_col, cat_col, ord_col, mean_col, freq_col, uknw_col, value):
    data.drop_duplicates(inplace=True)
    data, label = extract_label(data, label_column)
    imputed_data = data_preprocessing_impute(data)
    preprocessed_data = data_preprocessing_encode(imputed_data)

    return preprocessed_data, label

# One function to prepare the dataset for training
def prepare_dataset(data, label):
    X = np.array(data)
    y = np.array(label)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

# One function for training (typically applies up to 5 different methods for binary classification)
def training_validating_model(X_train, y_train, X_test, y_test, classifier):
    
    clf = classifier.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

# One function to display all the results in a convenient form for comparison
