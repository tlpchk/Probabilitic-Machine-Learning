from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics as sk_mtr
from math import sqrt

from src.data import split_dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def test(predicted, actual):
    return sk_mtr.classification_report(y_true=actual, y_pred=predicted, output_dict=True )

def test_cross(model, n_splits, df, class_column='class'):

    X = df.drop([class_column], axis=1)
    y = df.loc[:, df.columns == class_column]

    kf = StratifiedKFold(n_splits=n_splits)
    results = []
    for train_index, test_index in kf.split(X,y):
        train_X = pd.DataFrame(X.iloc[train_index, :].values, columns=X.columns).astype('category')
        train_y = pd.DataFrame(y.iloc[train_index, :].values, columns=y.columns).astype('category')

        model.fit(X=train_X, y=train_y)

        test_X = X.iloc[test_index, :]
        test_y = y.iloc[test_index, :][class_column]

        predicted_y = model.predict(X=test_X)

        res = test(predicted_y, test_y)

        results.append(res)

    results_dict = process_results(results)
    return results_dict

def test_split(model, n_splits, df, class_column='class'):

    results = []

    for i in range(n_splits):
        data = split_dataset(df, 0.8)

        train_X=data['train']['X']
        train_y=data['train']['y']

        test_X=data['test']['X']
        test_y=data['test']['y']

        model.fit(X=train_X, y=train_y)

        predicted_y = model.predict(X=test_X)

        res = test(predicted_y, test_y)
        results.append(res)

    results_dict = process_results(results)
    return results_dict

def process_results(results):

    avg_fscore = [0, 0]
    avg_prec = [0, 0]
    avg_rec = [0, 0]
    avg_acc = 0

    for res in results:
        avg_fscore[0] += res['0']['f1-score'] / len(results)
        avg_fscore[1] += res['1']['f1-score'] / len(results)
        avg_prec[0] += res['0']['precision'] / len(results)
        avg_prec[1] += res['1']['precision'] / len(results)
        avg_rec[0] += res['0']['recall'] / len(results)
        avg_rec[1] += res['1']['recall'] / len(results)
        avg_acc += res['accuracy'] / len(results)

    var_fscore = [0, 0]
    var_prec = [0, 0]
    var_rec = [0, 0]
    var_acc = 0

    for res in results:
        var_fscore[0] += (avg_fscore[0] - res['0']['f1-score'])**2 / len(results)
        var_fscore[1] += (avg_fscore[1] - res['1']['f1-score'])**2 / len(results)
        var_prec[0] += (avg_prec[0] - res['0']['precision'])**2 / len(results)
        var_prec[1] += (avg_prec[1] - res['1']['precision'])**2 / len(results)
        var_rec[0] += (avg_rec[0] - res['0']['recall'])**2 / len(results)
        var_rec[1] += (avg_rec[1] - res['1']['recall'])**2 / len(results)
        var_acc += (avg_acc - res['accuracy'])**2 / len(results)



    res_dic = {
        'avg_fscore' : [round(avg_fscore[0], 4), round(avg_fscore[1], 4)],
        'avg_precision' : [round(avg_prec[0], 4), round(avg_prec[1], 4)],
        'avg_recall' : [round(avg_rec[0], 4), round(avg_rec[1], 4)],
        'avg_accuracy'  :round(avg_acc, 4),

        'std_fscore' : [round(sqrt(var_fscore[0]), 4), round(sqrt(var_fscore[1]), 4)],
        'std_precision' : [round(sqrt(var_prec[0]), 4), round(sqrt(var_prec[1]), 4)],
        'std_recall' : [round(sqrt(var_rec[0]), 4), round(sqrt(var_rec[1]), 4)],
        'std_accuracy'  :round(sqrt(var_acc), 4)
    }

    return res_dic

def generate_plots(all_results):
    keys = []
    fscore = []
    precision = []
    recall = []
    accuracy = []

    for _, (model_name, v) in enumerate(all_results.items()):
        keys.append(model_name)
        fscore.append(v['avg_fscore'][0])
        precision.append(v['avg_precision'][0])
        recall.append(v['avg_recall'][0])
        accuracy.append(v['avg_accuracy'])

    plt.bar(keys, fscore)
    plt.xlabel("Models")
    plt.ylabel("Avg F1-Score")
    plt.savefig('avg_fscore_0.png')
    plt.show()

    plt.bar(keys, precision)
    plt.xlabel("Models")
    plt.ylabel("Avg Precision")
    plt.savefig('avg_precision_0.png')
    plt.show()

    plt.bar(keys, recall)
    plt.xlabel("Models")
    plt.ylabel("Avg Recall")
    plt.savefig('avg_recall_0.png')
    plt.show()

    plt.bar(keys, accuracy)
    plt.xlabel("Models")
    plt.ylabel("Avg Accuracy")
    plt.savefig('avg_accuracy.png')
    plt.show()

def generate_plots_std(all_results):
    keys = []
    fscore = []
    precision = []
    recall = []
    accuracy = []

    for _, (model_name, v) in enumerate(all_results.items()):
        keys.append(model_name)
        fscore.append(v['std_fscore'][0])
        precision.append(v['std_precision'][0])
        recall.append(v['std_recall'][0])
        accuracy.append(v['std_accuracy'])

    plt.bar(keys, fscore)
    plt.xlabel("Models")
    plt.ylabel("StdDev F1-Score")
    plt.savefig('std_fscore_0.png')
    plt.show()

    plt.bar(keys, precision)
    plt.xlabel("Models")
    plt.ylabel("StdDev Precision")
    plt.savefig('std_precision_0.png')
    plt.show()

    plt.bar(keys, recall)
    plt.xlabel("Models")
    plt.ylabel("StdDev Recall")
    plt.savefig('std_recall_0.png')
    plt.show()

    plt.bar(keys, accuracy)
    plt.xlabel("Models")
    plt.ylabel("StdDev Accuracy")
    plt.savefig('std_accuracy.png')
    plt.show()

