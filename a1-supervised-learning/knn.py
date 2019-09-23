import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

import dataset_processing as data_proc


def model_complexity_curve(X_train, y_train, hp, hp_vals, cv=None):
    df = pd.DataFrame(index=hp_vals, columns=['train', 'cv'])

    for hp_val in hp_vals:
        kwargs = {hp: hp_val, 'algorithm': 'kd_tree'}

        knnclf = KNeighborsClassifier(**kwargs)

        # train data
        knnclf.fit(X_train, y_train)
        train_score = knnclf.score(X_train, y_train)

        # get cv scores
        cross_vals = cross_val_score(knnclf, X_train, y_train, cv=cv)
        cv_mean = np.mean(cross_vals)

        df.loc[hp_val, 'train'] = train_score
        df.loc[hp_val, 'cv'] = cv_mean

    return pd.DataFrame(df, dtype='float')


def run_experiment(dataset_name, X_train, X_test, y_train, y_test, verbose=False, show_plots=False):
    # calculate model complexity scores for n_neighbors
    hp = 'n_neighbors'
    if dataset_name == 'online_shopping':
        hp_vals = np.arange(2, 30, 2)
    else:
        hp_vals = np.arange(2, 30, 2)
    if verbose:
        print(hp_vals)

    n_neighbors_mc = model_complexity_curve(
        X_train, y_train, hp, hp_vals, cv=data_proc.CV_VAL)
    n_neighbors_hp = n_neighbors_mc['cv'].idxmax()
    if verbose:
        print(n_neighbors_mc.head(10))
    if verbose:
        print(n_neighbors_mc.idxmax())

    data_proc.plot_model_complexity_charts(
        n_neighbors_mc['train'], n_neighbors_mc['cv'],
        dataset_name + ': MCC for ' + hp, hp)
    if show_plots:
        plt.show()

    plt.savefig('graphs/knn_mcc_' + hp + '_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    # calculate model complexity scores for leaf_size
    hp = 'leaf_size'
    hp_vals = np.arange(5, 100, 5)  # this should vary for each hyperparameter

    leaf_size_mc = model_complexity_curve(
        X_train, y_train, hp, hp_vals, cv=data_proc.CV_VAL)
    leaf_size_hp = leaf_size_mc['cv'].idxmax()
    if verbose:
        print(leaf_size_mc.head(10))
    if verbose:
        print(leaf_size_mc.idxmax())

    data_proc.plot_model_complexity_charts(
        leaf_size_mc['train'], leaf_size_mc['cv'],
        dataset_name + ': MCC for ' + hp, hp)
    if show_plots:
        plt.show()

    plt.savefig('graphs/knn_mcc_' + hp + '_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    # instantiate adaboost classifier
    knnclf = KNeighborsClassifier(
        n_neighbors=n_neighbors_hp, leaf_size=leaf_size_hp, algorithm='kd_tree')

    # calculate and print learning curves
    train_sizes = np.linspace(0.1, 0.9, 9)
    data_proc.plot_learning_curve(
        knnclf, dataset_name + ': learning curves',
        X_train, y_train, cv=data_proc.CV_VAL, train_sizes=train_sizes)
    if show_plots:
        plt.show()

    plt.savefig('graphs/knn_lc_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    knnclf.fit(X_train, y_train)

    train_score = data_proc.model_train_score(knnclf, X_train, y_train)
    test_score = data_proc.model_test_score(knnclf, X_test, y_test)
    print("KNeighborsClassifier training set score for " + dataset_name + ": ", train_score)
    print("KNeighborsClassifier holdout set score for " + dataset_name + ": ", test_score)


def abalone(verbose=False, show_plots=False):
    X_train, X_test, y_train, y_test = data_proc.process_abalone()

    run_experiment(
        'abalone', X_train, X_test, y_train, y_test,
        verbose=verbose, show_plots=show_plots)


def online_shopping(verbose=False, show_plots=False):
    X_train, X_test, y_train, y_test = data_proc.process_online_shopping()

    run_experiment(
        'online_shopping', X_train, X_test, y_train, y_test,
        verbose=verbose, show_plots=show_plots)


if __name__ == "__main__":
    abalone(verbose=True, show_plots=False)
    online_shopping(verbose=True, show_plots=False)
