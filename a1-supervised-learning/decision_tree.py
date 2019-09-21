import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
from time import time
import matplotlib.pyplot as plt

SEED_VAL = 313

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
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_times_df = train_times(estimator, X, y, train_sizes, cv=cv)

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
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    # plot times on the right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time(s)")
    ax2.plot(train_times_df['train_time'], 'o-', color='b', label="Training Time")
    ax2.plot(train_times_df['cv_time'], 'o-', color='y', label="CV Time")

    ax1.legend(loc="best")
    ax2.legend(loc="best")

    plt.tight_layout()
    return plt


def train_times(estimator, X, y, train_sizes, cv=None):
    train_times_df = pd.DataFrame(index=train_sizes, columns=['train_time', 'cv_time'])

    for train_size in train_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=SEED_VAL)

        # get train time
        train_t0 = time()
        estimator.fit(X_train, y_train)
        train_time = time() - train_t0

        # get cv time
        cv_t0 = time()
        cross_vals = np.mean(cross_val_score(estimator, X_train, y_train, cv=cv))
        cv_time = time() - cv_t0

        train_times_df.loc[train_size, 'train_time'] = train_time
        train_times_df.loc[train_size, 'cv_time'] = cv_time

    return train_times_df


def model_complexity_curve(X_train, y_train, X_test, y_test, hp, hp_vals, cv=None):
    df = pd.DataFrame(index=hp_vals, columns=['train', 'cv', 'test'])

    for hp_val in hp_vals:
        kwargs = { hp: hp_val }
        dtclf = DecisionTreeClassifier(**kwargs)

        # train data
        dtclf.fit(X_train, y_train)
        train_score = dtclf.score(X_train, y_train)

        # get cv scores
        cross_vals = np.mean(cross_val_score(dtclf, X_train, y_train, cv=cv))
        cv_mean = np.mean(cross_vals)

        # test data
        test_score = dtclf.score(X_test, y_test)

        df.loc[hp_val, 'train'] = train_score
        df.loc[hp_val, 'cv'] = cv_mean
        df.loc[hp_val, 'test'] = test_score
        # df.loc[hp_val, 'train_time'] = train_time
        # df.loc[hp_val, 'cv_time'] = cv_time

    return pd.DataFrame(df, dtype='float')


def model_complexity_charts(train_scores, test_scores, title, ylim=None):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid()


def main():
    verbose = True

    cv_val = 5
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED_VAL)

    # calculate model complexity scores for max_depth
    hp = 'max_depth'
    hp_vals = np.arange(3, 20)
    mc_curve = model_complexity_curve(X_train, y_train, X_test, y_test, hp, hp_vals, cv=cv_val)
    if verbose: print(mc_curve.head(20))

    #calculate model complexity scores for min_samples

    # instantiate decision tree
    if verbose: print(mc_curve.idxmax())
    dtclf = DecisionTreeClassifier(max_depth=mc_curve['test'].idxmax())

    # calculate and print learning curves
    train_sizes = np.linspace(.1, .9, 9)
    plot_learning_curve(dtclf, 'Learning Curves', X_train, y_train, cv=cv_val, train_sizes=train_sizes)
    # plt.show()


if __name__ == "__main__":
    main()
