import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
import matplotlib.pyplot as plt

def encode_data(df, cols):
    """
    Parameters:
        dataframe: list of
        cols (array-like): list of columns to encode

    """
    # encode
    l_enc = LabelEncoder()
    transformed = l_enc.fit_transform(df[[cols]])
    print(transformed)

    oh_enc = OneHotEncoder()

    # train/test split

    # cross validation

    # fit

if __name__ == "__main__":
    seed_val = 313
    abalone_names = [
        'sex', 'length', 'diameter', 'height', 'whole_weight',
        'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'
        ]
    df = pd.read_csv('./abalone.csv', names=abalone_names)
    df = df.dropna()

    # transform output into classification problem
    df.loc[df['rings'] < 9, 'rings'] = 1
    df.loc[(df['rings'] >= 9) & (df['rings'] <= 10), 'rings'] = 2
    df.loc[df['rings'] > 10, 'rings'] = 3

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X = pd.get_dummies(X)

    # split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed_val)

    # instantiate decision tree
    dtclf = DecisionTreeClassifier()

    outputs = dtclf.fit(X_train, y_train)

    # cross validation scores to find best coefficients
    scores = cross_val_score(dtclf, X_train, y=y_train)

    # calculate and print learning curves

    print(scores)