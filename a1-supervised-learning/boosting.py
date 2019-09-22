import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

import dataset_processing as data_proc


def model_complexity_curve(estimator, X_train, y_train, hp, hp_vals, cv=None):
    df = pd.DataFrame(index=hp_vals, columns=['train', 'cv'])

    for hp_val in hp_vals:
        kwargs = {
            'base_estimator': estimator,
            hp: hp_val,
            'random_state': data_proc.SEED_VAL}

        adaclf = AdaBoostClassifier(**kwargs)

        # train data
        adaclf.fit(X_train, y_train)
        train_score = adaclf.score(X_train, y_train)

        # get cv scores
        cross_vals = cross_val_score(adaclf, X_train, y_train, cv=cv)
        cv_mean = np.mean(cross_vals)

        df.loc[hp_val, 'train'] = train_score
        df.loc[hp_val, 'cv'] = cv_mean

    return pd.DataFrame(df, dtype='float')


def run_experiment(dt_estimator, dataset_name, X_train, X_test, y_train, y_test, verbose=False, show_plots=False):
    # calculate model complexity scores for n_estimators
    hp = 'n_estimators'
    if dataset_name == 'online_shopping':
        hp_vals = np.arange(5, 30, 5)
    else:
        hp_vals = np.arange(10, 60, 10)
    if verbose:
        print(hp_vals)

    n_estimators_mc = model_complexity_curve(
        dt_estimator, X_train, y_train, hp, hp_vals, cv=data_proc.CV_VAL)
    n_estimators_hp = n_estimators_mc['cv'].idxmax()
    if verbose:
        print(n_estimators_mc.head(10))
    if verbose:
        print(n_estimators_mc.idxmax())

    data_proc.plot_model_complexity_charts(
        n_estimators_mc['train'], n_estimators_mc['cv'],
        dataset_name + ': MCC for ' + hp, hp)
    if show_plots:
        plt.show()

    plt.savefig('graphs/ada_mcc_' + hp + '_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    # calculate model complexity scores for learning_rate
    hp = 'learning_rate'
    hp_vals = np.logspace(-3, 1, base=10.0, num=5)  # this should vary for each hyperparameter

    learning_rate_mc = model_complexity_curve(
        dt_estimator, X_train, y_train, hp, hp_vals, cv=data_proc.CV_VAL)
    learning_rate_hp = learning_rate_mc['cv'].idxmax()
    if verbose:
        print(learning_rate_mc.head(10))
    if verbose:
        print(learning_rate_mc.idxmax())

    data_proc.plot_model_complexity_charts(
        learning_rate_mc['train'], learning_rate_mc['cv'],
        dataset_name + ': MCC for ' + hp, hp, xscale_type='log')
    if show_plots:
        plt.show()

    plt.savefig('graphs/ada_mcc_' + hp + '_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    # instantiate adaboost classifier
    adaclf = AdaBoostClassifier(
        dt_estimator, n_estimators=n_estimators_hp, learning_rate=learning_rate_hp,
        random_state=data_proc.SEED_VAL)

    # calculate and print learning curves
    train_sizes = np.linspace(0.1, 0.9, 9)
    data_proc.plot_learning_curve(
        adaclf, dataset_name + ': learning curves',
        X_train, y_train, cv=data_proc.CV_VAL, train_sizes=train_sizes)
    if show_plots:
        plt.show()

    plt.savefig('graphs/ada_lc_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    adaclf.fit(X_train, y_train)

    train_score = data_proc.model_train_score(adaclf, X_train, y_train)
    test_score = data_proc.model_test_score(adaclf, X_test, y_test)
    print("AdaBoostClassifier training set score for " + dataset_name + ": ", train_score)
    print("AdaBoostClassifier holdout set score for " + dataset_name + ": ", test_score)


def abalone(dt_estimator, verbose=False, show_plots=False):
    X_train, X_test, y_train, y_test = data_proc.process_abalone()

    run_experiment(
        dt_estimator, 'abalone', X_train, X_test, y_train, y_test,
        verbose=verbose, show_plots=show_plots)


def online_shopping(dt_estimator, verbose=False, show_plots=False):
    X_train, X_test, y_train, y_test = data_proc.process_online_shopping()

    run_experiment(
        dt_estimator, 'online_shopping', X_train, X_test, y_train, y_test,
        verbose=verbose, show_plots=show_plots)


if __name__ == "__main__":
    dt_abalone = DecisionTreeClassifier(max_depth=6, random_state=data_proc.SEED_VAL)
    dt_online_shopping = DecisionTreeClassifier(max_depth=5, random_state=data_proc.SEED_VAL)

    abalone(dt_abalone, verbose=True, show_plots=False)
    online_shopping(dt_online_shopping, verbose=True, show_plots=False)
