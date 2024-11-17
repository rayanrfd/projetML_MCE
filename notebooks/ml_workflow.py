# Importing the basic libraries
import numpy as np
import pandas as pd

# Importing preprocessers
from sklearn.model_selection import train_test_split, cross_val_score

# Importing the classifiers
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier

# One function for pre-processing

## Kidney dataset '?' handling
def question_mark_handling(data, cols):
    for col in cols:
        filter = data[col].str.contains('?')
        data[filter].loc[:, col] = np.nan
    return data

##### classification, cad, dm,
##### pcv, wc, rc


## Divides the dataset between the features and the label
def extract_label(data, column):
    label = data[column]
    data = data.drop(column, axis=1)
    return data, label

## 1st step of the data preprocessing : Imputing the missing values
def data_preprocessing_impute(data): #mean_col, freq_col, uknw_col, value):
    
    # We take the columns with numerical values
    mean_col = data.select_dtypes(include=['int64', 'float64']).columns.to_list()
    # We replace the missing values for each columns by its mean
    if mean_col is not None:
        data[mean_col] = data[mean_col].fillna(data[mean_col].mean())

    # We take the columns with categorical values
    freq_col = data.select_dtypes(include=['object']).columns.to_list()
    # We replace the missing values for each columns by its most frequent value
    if freq_col is not None:
        data[freq_col] = data[freq_col].fillna(data[freq_col].mode())

    return data

## 2nd step : Scaling the numerical features and encoding the categorical ones
def data_preprocessing_encode(data): #, num_col, cat_col, ord_col):
     
    # We take the columns with numerical values
    num_col = data.select_dtypes(include=['int64', 'float64']).columns.to_list()
    # We scale them (new range [0;1])
    if num_col is not None:
        data[num_col] = (data[num_col] - data[num_col].min()) / (data[num_col].max() - data[num_col].min())

    # We take the columns with categorical values
    cat_col = data.select_dtypes(include=['object']).columns.to_list()
    # We one-hot encode the columns
    if cat_col is not None: 
        data = pd.get_dummies(data, columns=cat_col, dtype=int)

    return data

## We combine the two preprocessing steps into one function
def data_preprocessing(data, label_column): #, num_col, cat_col, ord_col, mean_col, freq_col, uknw_col, value):
    data.drop_duplicates(inplace=True)
    data, label = extract_label(data, label_column)
    imputed_data = data_preprocessing_impute(data)
    preprocessed_data = data_preprocessing_encode(imputed_data)

    return preprocessed_data, label

# One function to prepare the dataset for training
def prepare_dataset(data, label):
    # We convert the data from the DataFrame type to the array type
    # Only arrays can be handled by the Scikit-Learn library
    X = np.array(data)
    y = np.array(label)
    # We split the dataset between a train set, a validation set and a test set
    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.33, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# One function for training (typically applies up to 5 different methods for binary classification)
def training_validating_model(X_train, X_val, X_test, y_train, y_val, y_test, classifier):
    
    clf = classifier.fit(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    val_score = cross_val_score(classifier, X_val, y_val)
    return test_score, val_score

# One function to display all the results in a convenient form for comparison
def display_results():
    pass
