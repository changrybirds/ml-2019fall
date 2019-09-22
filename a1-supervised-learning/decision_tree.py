import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

import dataset_processing as data_proc

def model_complexity_curve(X_train, X_test, y_train, y_test, hp, hp_vals, cv=None):
    df = pd.DataFrame(index=hp_vals, columns=['train', 'test'])

    for hp_val in hp_vals:
        kwargs = {
            hp: hp_val,
            'random_state': data_proc.SEED_VAL
            }
        dtclf = DecisionTreeClassifier(**kwargs)

        # train data
        dtclf.fit(X_train, y_train)
        train_score = dtclf.score(X_train, y_train)

        # get cv scores
        # cross_vals = np.mean(cross_val_score(dtclf, X_train, y_train, cv=cv))
        # cv_mean = np.mean(cross_vals)

        # test data
        test_score = dtclf.score(X_test, y_test)

        df.loc[hp_val, 'train'] = train_score
        # df.loc[hp_val, 'cv'] = cv_mean
        df.loc[hp_val, 'test'] = test_score
        # df.loc[hp_val, 'train_time'] = train_time
        # df.loc[hp_val, 'cv_time'] = cv_time

    return pd.DataFrame(df, dtype='float')


def run_experiment(dataset_name, X_train, X_test, y_train, y_test, verbose=False, show_plots=False):
    # calculate model complexity scores for max_depth
    hp = 'max_depth'
    hp_vals = np.arange(3, 20)
    max_depth_mc = model_complexity_curve(X_train, X_test, y_train, y_test, hp, hp_vals, cv=data_proc.CV_VAL)
    max_depth_hp = max_depth_mc['test'].idxmax()
    if verbose: print(max_depth_mc.head(10))
    if verbose: print(max_depth_mc.idxmax())

    data_proc.plot_model_complexity_charts(max_depth_mc['train'], max_depth_mc['test'], 'MCC for ' + hp, hp)
    if show_plots: plt.show()

    plt.savefig('dt_mcc_' + hp + '_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    # calculate model complexity scores for max_features
    hp = 'max_features'
    hp_vals = np.arange(1, X_train.shape[1])
    max_features_mc = model_complexity_curve(X_train, X_test, y_train, y_test, hp, hp_vals, cv=data_proc.CV_VAL)
    max_features_hp = max_features_mc['test'].idxmax()
    if verbose: print(max_features_mc.head(10))
    if verbose: print(max_features_mc.idxmax())

    data_proc.plot_model_complexity_charts(max_features_mc['train'], max_features_mc['test'], 'MCC for ' + hp, hp)
    if show_plots: plt.show()

    plt.savefig('dt_mcc_' + hp + '_' + dataset_name + '.png')
    plt.clf()
    plt.close()

    # instantiate decision tree
    dtclf = DecisionTreeClassifier(max_depth=max_depth_hp, max_features=max_features_hp, random_state=data_proc.SEED_VAL)

    # calculate and print learning curves
    train_sizes = np.linspace(0.1, 0.9, 9)
    data_proc.plot_learning_curve(dtclf, 'Learning Curves', X_train, y_train, cv=data_proc.CV_VAL, train_sizes=train_sizes)
    if show_plots: plt.show()

    plt.savefig('dt_lc_' + dataset_name + '.png')
    plt.clf()
    plt.close()


def abalone(verbose=False, show_plots=False):
    abalone_names = [
        'sex', 'length', 'diameter', 'height', 'whole_weight',
        'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'
        ]
    df = pd.read_csv('./abalone.csv', header=None, names=abalone_names)
    df = df.dropna()

    X_train, X_test, y_train, y_test = data_proc.process_abalone(df)

    run_experiment('abalone', X_train, X_test, y_train, y_test, verbose=verbose, show_plots=show_plots)


def online_shopping(verbose=False, show_plots=False):
    df = pd.read_csv('./online_shoppers_intention.csv')
    df = df.dropna()

    X_train, X_test, y_train, y_test = data_proc.process_online_shopping(df)

    run_experiment('online_shopping', X_train, X_test, y_train, y_test, verbose=verbose, show_plots=show_plots)


if __name__ == "__main__":
    # abalone(verbose=False, show_plots=False)
    online_shopping(verbose=True, show_plots=False)
