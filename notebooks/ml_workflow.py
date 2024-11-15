import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, Normalizer, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier


# One function for pre-processing
def extract_label(data, column):
    data = data.drop(column, axis=1)
    label = data[column]
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
        data[num_col] = (data[num_col] - data[num_col].mean()) / data[num_col].std()

    cat_col = data.select_dtypes(include=['object']).columns.to_list()
    if cat_col is not None: 
        data = pd.get_dummies(data, columns=cat_col, dtype=int)

    return data

def data_preprocessing(data, label_column): #, num_col, cat_col, ord_col, mean_col, freq_col, uknw_col, value):
    data, label = extract_label(data, label_column)
    imputed_data = data_preprocessing_impute(data)
    preprocessed_data = data_preprocessing_encode(imputed_data)

    return preprocessed_data, label

# One function to prepare the dataset for training
def prepare_dataset(data, label):
    X, y = np.array(data), np.array(label)
    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.1)

    return X_train, X_val, X_test, y_train, y_val, y_test

# One function for training (typically applies up to 5 different methods for binary classification)
def training_validating_model(X_train, y_train, X_test, y_test, classifier):
    
    clf = classifier.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

# One function to display all the results in a convenient form for comparison
