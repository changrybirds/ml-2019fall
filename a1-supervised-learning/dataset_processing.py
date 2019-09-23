import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve

from time import time

SEED_VAL = 313
CV_VAL = 5
HOLDOUT_SIZE = 0.3


# not needed for these datasets
def encode_data(df, cols):
    """
    Parameters:
        dataframe: list of
        cols (array-like): list of columns to encode

    """
    # encode
    l_enc = LabelEncoder()
    transformed = l_enc.fit_transform(df[[cols]])

    oh_enc = OneHotEncoder()
    encoded = oh_enc.fit_transform(transformed)

    return encoded


def scale_data(X):
    X = X.values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)

    return scaled


def process_abalone():
    abalone_names = [
        'sex', 'length', 'diameter', 'height', 'whole_weight',
        'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'
        ]
    df = pd.read_csv('./abalone.csv', header=None, names=abalone_names)
    df = df.dropna()

    # transform output into classification problem
    df.loc[df['rings'] < 9, 'rings'] = 1
    df.loc[(df['rings'] >= 9) & (df['rings'] <= 10), 'rings'] = 2
    df.loc[df['rings'] > 10, 'rings'] = 3

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # encode data
    X = pd.get_dummies(X)

    return train_test_split(X, y, test_size=HOLDOUT_SIZE, random_state=SEED_VAL)


def process_online_shopping():
    df = pd.read_csv('./online_shoppers_intention.csv')
    df = df.dropna()

    df = df.astype({
        'OperatingSystems': 'str',
        'Browser': 'str',
        'Region': 'str',
        'TrafficType': 'str'
    })

    X = pd.concat((df.iloc[:, :-6], df.iloc[:, -5:]), axis=1)
    y = df.iloc[:, -6]

    # combine low instance browsers into one class
    mask = y.isin(['1', '2'])
    y = y.where(mask, other='3')
    # y = y.astype(str)

    # encode data
    X = pd.get_dummies(X)

    return train_test_split(X, y, test_size=HOLDOUT_SIZE, random_state=SEED_VAL)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    # adapted from sklearn documentation:
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    fig, ax1 = plt.subplots()
    plt.grid()
    plt.title(title)
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.set_xlabel("Training examples")
    ax1.set_ylabel("Score")

    train_sizes_abs, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=CV_VAL, n_jobs=n_jobs,
        train_sizes=train_sizes, shuffle=True, random_state=SEED_VAL)

    train_times_df = train_times(estimator, X, y, train_sizes, cv=CV_VAL)

    # collapse cv folds into a single value
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # plot scores on left axis
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    line1 = ax1.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
    line2 = ax1.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")

    # plot times on the right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time(s)")
    line3 = ax2.plot(train_times_df['train_time'], 'o-', color='b', label="Training Time")
    line4 = ax2.plot(train_times_df['cv_time'], 'o-', color='y', label="CV Time")

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.tight_layout()
    return plt


def train_times(estimator, X, y, train_sizes, cv=None):
    train_times_df = pd.DataFrame(index=train_sizes, columns=['train_time', 'cv_time'])
    for train_size in train_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=SEED_VAL)

        # get train time
        train_t0 = time()
        estimator.fit(X_train, y_train)
        train_time = time() - train_t0

        # get cv time
        cv_t0 = time()
        cross_vals = cross_val_score(estimator, X_train, y_train, cv=CV_VAL)
        cv_time = time() - cv_t0

        train_times_df.loc[train_size, 'train_time'] = train_time
        train_times_df.loc[train_size, 'cv_time'] = cv_time

    return train_times_df


def plot_model_complexity_charts(train_scores, test_scores, title, hp_name, ylim=None, xscale_type=None, basex=None):
    plt.figure()
    plt.title(title)
    plt.xlabel(hp_name)
    plt.ylabel("Score")

    if ylim is not None:
        plt.ylim(*ylim)
    if xscale_type is not None:
        plt.xscale(xscale_type, basex=basex)
    plt.grid()

    plt.plot(train_scores, 'o-', color="r", label="Training score")
    plt.plot(test_scores, 'o-', color="g", label="CV score")
    plt.legend(loc='best')

    return plt


def plot_iterative_lc(df, title, max_iter_range, ylim=None):
    fig, ax1 = plt.subplots()
    plt.grid()
    plt.title(title)
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.set_xlabel("max_iterations")
    ax1.set_ylabel("Score")

    train_scores = df['train']
    train_scores_std = np.std(df['train'])
    cv_scores = df['cv']
    cv_scores_std = np.std(df['cv'])

    # plot scores on left axis
    # ax1.fill_between(max_iter_range, train_scores - train_scores_std,
    #                  train_scores + train_scores_std, alpha=0.1, color="r")
    # ax1.fill_between(max_iter_range, cv_scores - cv_scores_std,
    #                  cv_scores + cv_scores_std, alpha=0.1, color="g")
    line1 = ax1.plot(max_iter_range, train_scores, 'o-', color="r",
             label="Training score")
    line2 = ax1.plot(max_iter_range, cv_scores, 'o-', color="g",
             label="CV score")

    # plot times on the right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time(s)")
    line3 = ax2.plot(df['train_time'], 'o-', color='b', label="Training Time")
    line4 = ax2.plot(df['cv_time'], 'o-', color='y', label="CV Time")

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")

    plt.tight_layout()
    return plt


def model_train_score(estimator, X_train, y_train):
    return estimator.score(X_train, y_train)

def model_test_score(estimator, X_test, y_test):
    return estimator.score(X_test, y_test)
