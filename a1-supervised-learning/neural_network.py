import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from time import time

import dataset_processing as data_proc


def model_complexity_curve(X_train, y_train, hp, hp_vals, cv=None):
    df = pd.DataFrame(index=hp_vals, columns=['train', 'cv'])

    for hp_val in hp_vals:
        kwargs = {
            hp: hp_val,
            'random_state': data_proc.SEED_VAL}

        mlpclf = MLPClassifier(**kwargs)

        # train data
        mlpclf.fit(X_train, y_train)
        train_score = mlpclf.score(X_train, y_train)

        # get cv scores
        cross_vals = cross_val_score(mlpclf, X_train, y_train, cv=cv)
        cv_mean = np.mean(cross_vals)

        df.loc[hp_val, 'train'] = train_score
        df.loc[hp_val, 'cv'] = cv_mean

    return pd.DataFrame(df, dtype='float')


def plot_iterative_lc(estimator, title, X, y, max_iter_range, ylim=None, cv=None,
                      n_jobs=None):
    df = pd.DataFrame(index=max_iter_range, columns=['train', 'cv', 'train_time', 'cv_time'])
    for i in max_iter_range:
        kwargs = {
            max_iter: i,
            'random_state': data_proc.SEED_VAL}

        mlpclf = MLPClassifier(**kwargs)

        # train data
        train_t0 = time()
        mlpclf.fit(X_train, y_train)
        train_time = time() - train_t0
        train_score = mlpclf.score(X_train, y_train)

        # get cv scores
        cv_t0 = time()
        cross_vals = cross_val_score(mlpclf, X_train, y_train, cv=cv)
        cv_time = time() - cv_t0
        cv_mean = np.mean(cross_vals)

        df.loc[i, 'train'] = train_score
        df.loc[i, 'cv'] = cv_mean
        df.loc[i, 'train_time'] = train_time
        df.loc[i, 'cv_time'] = cv_time

    fig, ax1 = plt.subplots()
    plt.grid()
    plt.title(title)
    if ylim is not None:
        ax1.set_ylim(*ylim)
    ax1.set_xlabel("Training examples")
    ax1.set_ylabel("Score")

    train_scores_mean = np.mean(df['train'], axis=1)
    train_scores_std = np.std(df['train'], axis=1)
    cv_scores_mean = np.mean(df['cv'], axis=1)
    cv_scores_std = np.std(df['cv'], axis=1)

    # plot scores on left axis
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    ax1.fill_between(train_sizes, cv_scores_mean - cv_scores_std,
                     cv_scores_mean + cv_scores_std, alpha=0.1, color="g")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    ax1.plot(train_sizes, cv_scores_mean, 'o-', color="g",
             label="CV score")

    # plot times on the right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Time(s)")
    ax2.plot(df['train_time'], 'o-', color='b', label="Training Time")
    ax2.plot(df['cv_time'], 'o-', color='y', label="CV Time")

    ax1.legend(loc="best")
    ax2.legend(loc="best")

    plt.tight_layout()
    return plt


def run_experiment(dataset_name, X_train, X_test, y_train, y_test, verbose=False, show_plots=False):
    # calculate model complexity scores for hidden_layer_sizes
    num_features = X_train.shape[1]
    hp = 'hidden_layer_sizes'

    # using 1 hidden layer - problems shouldn't be so complex as to require more
    hp_vals = (np.arange(1, num_features),)  # this should vary for each hyperparameter
    hidden_layer_sizes_mc = model_complexity_curve(
        X_train, y_train, hp, hp_vals, cv=data_proc.CV_VAL)
    hidden_layer_sizes_hp = hidden_layer_sizes_mc['cv'].idxmax()

    if verbose:
        print(hidden_layer_sizes_mc.head(10))
    if verbose:
        print(hidden_layer_sizes_mc.idxmax())

    data_proc.plot_model_complexity_charts(
        hidden_layer_sizes_mc['train'], hidden_layer_sizes_mc['cv'],
        dataset_name + ': MCC for ' + hp, hp)

    if show_plots:
        plt.show()

    plt.savefig('graphs/nn_mcc_' + hp + '_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    # calculate model complexity scores for learning_rate_init
    hp = 'learning_rate_init'
    hp_vals = np.logspace(-5, 0, base=10.0)    # this should vary for each hyperparameter
    learning_rate_init_mc = model_complexity_curve(
        X_train, y_train, hp, hp_vals, cv=data_proc.CV_VAL)
    learning_rate_init_hp = learning_rate_init_mc['cv'].idxmax()

    if verbose:
        print(learning_rate_init_mc.head(10))
    if verbose:
        print(learning_rate_init_mc.idxmax())

    data_proc.plot_model_complexity_charts(
        learning_rate_init_mc['train'], learning_rate_init_mc['cv'],
        dataset_name + ': MCC for ' + hp, hp)

    if show_plots:
        plt.show()

    plt.savefig('graphs/nn_mcc_' + hp + '_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    # instantiate decision tree
    mlpclf = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes_hp, learning_rate_init=learning_rate_init_hp,
        random_state=data_proc.SEED_VAL)

    # calculate and print learning curves, use max_iter as x-axis
    max_iter_range = np.arange(50, 250, 10)
    data_proc.plot_iterative_lc(
        mlpclf, dataset_name + ': learning curves',
        X_train, y_train, max_iter_range=max_iter_range, cv=data_proc.CV_VAL)

    if show_plots:
        plt.show()

    plt.savefig('graphs/nn_lc_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    test_scores = data_proc.model_test_score(mlpclf, X_test, y_test)
    print("MLPClassifier holdout set score for " + dataset_name + ": ", test_scores)


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
    abalone(verbose=False, show_plots=False)
    online_shopping(verbose=False, show_plots=False)
