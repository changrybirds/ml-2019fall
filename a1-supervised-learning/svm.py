import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from time import time

import dataset_processing as data_proc

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def model_complexity_curve(kernel, X_train, y_train, max_iter, hp, hp_vals, cv=None):
    df = pd.DataFrame(index=hp_vals, columns=['train', 'cv'])

    for hp_val in hp_vals:
        kwargs = {
            'kernel': kernel,
            'gamma': 'auto',
            hp: hp_val,
            'max_iter': max_iter,
            'random_state': data_proc.SEED_VAL}

        svmclf = SVC(**kwargs)

        # train data
        svmclf.fit(X_train, y_train)
        train_score = svmclf.score(X_train, y_train)

        # get cv scores
        cross_vals = cross_val_score(svmclf, X_train, y_train, cv=cv)
        cv_mean = np.mean(cross_vals)

        df.loc[hp_val, 'train'] = train_score
        df.loc[hp_val, 'cv'] = cv_mean

    return pd.DataFrame(df, dtype='float')


@ignore_warnings(category=ConvergenceWarning)
def svm_iterative_lc(kernel, X, y, max_iter_range, cv=None):
    df = pd.DataFrame(index=max_iter_range, columns=['train', 'cv', 'train_time', 'cv_time'])
    for i in max_iter_range:
        kwargs = {
            'kernel': kernel,
            'gamma': 'auto',
            'max_iter': i,
            'random_state': data_proc.SEED_VAL}

        svmclf = SVC(**kwargs)

        # train data
        train_t0 = time()
        svmclf.fit(X, y)
        train_time = time() - train_t0
        train_score = svmclf.score(X, y)

        # get cv scores
        cv_t0 = time()
        cross_vals = cross_val_score(svmclf, X, y, cv=cv)
        cv_time = time() - cv_t0
        cv_mean = np.mean(cross_vals)

        df.loc[i, 'train'] = train_score
        df.loc[i, 'cv'] = cv_mean
        df.loc[i, 'train_time'] = train_time
        df.loc[i, 'cv_time'] = cv_time

    return df.astype('float64')


@ignore_warnings(category=ConvergenceWarning)
def run_experiment(kernel, dataset_name, X_train, X_test, y_train, y_test, verbose=False, show_plots=False):
    # calculate and print learning curves, use max_iter as x-axis
    if dataset_name == 'online_shopping':
        # max_iter_range = np.arange(100, 500, 50)
        max_iter_range = np.arange(1000, 2000, 100)
    else:
        max_iter_range = np.arange(1000, 2000, 100)

    lc_df = svm_iterative_lc(kernel, X_train, y_train, max_iter_range, cv=data_proc.CV_VAL)
    if verbose:
        print(lc_df.head(10))

    max_iter_hp = lc_df['cv'].idxmax()
    if verbose:
        print(lc_df.idxmax())

    data_proc.plot_iterative_lc(
        lc_df, dataset_name + ' - ' + kernel + ': learning curves (iterations)',
        max_iter_range=max_iter_range)

    if show_plots:
        plt.show()

    plt.savefig('graphs/svm_' + kernel + '_iterlc_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    # calculate model complexity scores for C
    hp = 'C'
    if kernel == 'linear':
        # hp_vals = np.logspace(-13, -8, base=2.0, num=6)
        hp_vals = np.logspace(-3, 3, base=2.0, num=7)
    else:
        hp_vals = np.logspace(-5, 3, base=2.0, num=9)  # this should vary for each hyperparameter

    if verbose:
        print(hp_vals)
    C_mc = model_complexity_curve(
        kernel, X_train, y_train, max_iter_hp, hp, hp_vals, cv=data_proc.CV_VAL)
    C_hp = C_mc['cv'].idxmax()

    if verbose:
        print(C_mc.head(10))
    if verbose:
        print(C_mc.idxmax())

    data_proc.plot_model_complexity_charts(
        C_mc['train'], C_mc['cv'],
        dataset_name + ' - ' + kernel + ': MCC for ' + hp, hp, xscale_type='log', basex=2)

    if show_plots:
        plt.show()

    plt.savefig('graphs/svm_' + kernel + '_mcc_' + hp + '_' + dataset_name + '.png')
    plt.clf()
    plt.close()


    # instantiate SVC
    svmclf = SVC(
        kernel=kernel, C=C_hp, gamma='auto', max_iter=max_iter_hp, random_state=data_proc.SEED_VAL)

    svmclf.fit(X_train, y_train)

    # calculate and print learning curves
    train_sizes = np.linspace(0.1, 0.9, 9)
    data_proc.plot_learning_curve(
        svmclf, dataset_name + ' - ' + kernel + ': learning curves',
        X_train, y_train, cv=data_proc.CV_VAL, train_sizes=train_sizes)
    if show_plots:
        plt.show()

    plt.savefig('graphs/svm_' + kernel + '_lc_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    svmclf.fit(X_train, y_train)

    train_score = data_proc.model_train_score(svmclf, X_train, y_train)
    test_score = data_proc.model_test_score(svmclf, X_test, y_test)
    print("SVC " + kernel + " training set score for " + dataset_name + ": ", train_score)
    print("SVC " + kernel + " holdout set score for " + dataset_name + ": ", test_score)


def abalone(kernels, verbose=False, show_plots=False):
    X_train, X_test, y_train, y_test = data_proc.process_abalone()

    for col in X_train.columns:
        X_train[col] = data_proc.scale_data(X_train[col])
        X_test[col] = data_proc.scale_data(X_test[col])

    for kernel in kernels:
        run_experiment(
            kernel, 'abalone', X_train, X_test, y_train, y_test,
            verbose=verbose, show_plots=show_plots)


def online_shopping(kernels, verbose=False, show_plots=False):
    X_train, X_test, y_train, y_test = data_proc.process_online_shopping()
    scalable_cols = [
        'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
        'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

    for col in scalable_cols:
        X_train[col] = data_proc.scale_data(X_train[col])
        X_test[col] = data_proc.scale_data(X_test[col])

    for kernel in kernels:
        run_experiment(
            kernel, 'online_shopping', X_train, X_test, y_train, y_test,
            verbose=verbose, show_plots=show_plots)


if __name__ == "__main__":
    kernels = ['linear', 'rbf']
    abalone(kernels, verbose=True, show_plots=False)
    online_shopping(kernels, verbose=True, show_plots=False)
