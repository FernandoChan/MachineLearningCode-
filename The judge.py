# -*- coding: utf-8 -*-
"""
 * file: search
 * Creator: 陈效威
 * Date: 2019-04-22
 * Time: 11:39
"""

import logging

logger = logging.getLogger('judge')

ID = 201630588238
NAME = "陈效威"
print('------------------------------------------------------------------------------')
print('Author；{0} , ID: {1} '.format(NAME, ID))
dataset_num = ID  % 349
print('Using dataset NO. {0}, ID % 349 = {1} % 349'.format(dataset_num, ID))

algr_str = 'LMNBCTRAGP'
maps = {'L': '线性判别分析LDA',
        'M': '支持向量机',
        'N': '最近邻分类器',
        'B': '朴素贝叶斯；',
        'C': '决策树C4.5',
        'T': '分类与回归树',
        'R': '随机森林',
        'A': 'Ada Boost',
        'G': 'Gradient Tree Boosting',
        'P': '标签传播', }

name = 'CXW'

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00404/OBS-Network-DataSet_2_Aug27.arff"


def _getDatabaseOnline(url):
    import requests
    from scipy.io import arff

    with requests.get(url) as response:
        try:
            return  arff.loadarff(response.content)
        except Exception as e:
            logger.exception(e.args)


# data = _getDatabaseOnline(url)


def judge(name, algr):
    name_list = list_2ord(list(name.lower()))
    algr_list = list_2ord(list(algr.lower()))

    if len(name_list) == 2:
        name_list.append((name_list[-1] + 1) % 26)
    elif len(name_list) > 3:
        name_list = name_list[:3]
    if len(algr_list) <= 3:
        return algr

    res = []
    if name_list.__len__() == 3:
        for i in range(algr_list.__len__() - name_list.__len__() + 1):
            sum = 0
            for j in range(name_list.__len__()):
                sum += pow(abs(algr_list[i + j] - name_list[j]), 2)
            res.append(sum)
        inx = res.index(max(res))
        return algr[inx: inx + 3]


def list_2ord(obj):
    try:
        if (isinstance(obj, list)):
            res = [ord(i) - 96 for i in obj if ord(i) < 122 and ord(i) >= 97]
            return res
        elif (isinstance(obj, str)):
            return list_2ord(list(obj.lower()))
        else:
            raise TypeError('invalid sequence %s' % obj)
    except TypeError as e:
        logger.exception(e)


print("My algorithm sequence is : {0}".format(judge(name, algr_str)))
print("Using algorithm: {0}, respectively".format([maps[i] for i in judge(name, algr_str)]))
print("------------------------------------------------------------------------------")


import logging
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC, \
    AdaBoostClassifier as AdaBC, ExtraTreesClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import KFold,train_test_split, cross_val_score , GridSearchCV

def _classification_dataset(name):
    data = arff.loadarff(name)
    df = pd.DataFrame(data[0])
    print("Dataset Info: ")
    print("Dataset has {0} features ,  ")

    print("------------------------------------------------------------------------------")

    df = df.dropna()
    node_status_map = {b'B': 0, b'NB': 1, b"'P NB'": 2}
    df['Node Status'] = df['Node Status'].dropna().map(node_status_map).astype(int)  # encoding

    class_map = {b"'NB-No Block'": 1, b"'No Block'": 2, b'Block': 0, b'NB-Wait': 3}
    df['Class'] = df['Class'].dropna().map(class_map).astype(int)  # encoding

    X = df.drop(['Class', ], axis=1).values

    y = df.iloc[:, -1].values  # array

    return X, y


try:
    X, y = _classification_dataset('OBS-Network-DataSet_2_Aug27.arff')
except FileNotFoundError as e:
    logger.exception(e.args)

one_hot = OneHotEncoder()
label_encoder = LabelEncoder()
label_encoder.fit(y)
encode_y = label_encoder.transform(y)
dummy_y = np_utils.to_categorical(encode_y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rfc_clf = RFC(n_estimators=7, criterion='gini', random_state=42, oob_score=False,
              max_features="auto", )
gbc_clf = GBC(random_state=42, )
abc_clf = AdaBC(random_state=42, )
extra_tree = ExtraTreesClassifier(n_estimators=10, random_state=42)
clf = [rfc_clf, gbc_clf, abc_clf, extra_tree]

params = {"n_estimators": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          "criterion":["entropy", "gini"],
          "n_jobs":[1,-1],
          "warm_start":[True,False],
          "max_depth" : list(range (10,100,10))
          }

grid = GridSearchCV(cv=10, estimator=rfc_clf, param_grid=params)
grid.fit(X_train, y_train)
print("grid score is {0}".format(grid.best_score_))
print("The grid best param is {0}".format(grid.best_params_))

best_forest = grid.best_estimator_
clf.append(best_forest)
sorted(grid.cv_results_.keys())

metric = [accuracy_score, mean_squared_error]
scaler = preprocessing.StandardScaler.fit(self=preprocessing.StandardScaler(), X=X_train)
X_trans = scaler.transform(X_train)
X_test_trans = scaler.transform(X_test)

for classifier in clf:
    classifier.fit(X_train, y_train)
    pred_train = classifier.predict(X_train)
    pred_test = classifier.predict(X_test)
    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    print("The {2} has {3} estimators , got a train score: {0:.4f}, test score: {1:.4f}".format(train_acc, test_acc,
                                                                                                classifier.__class__.__name__,
                                                                                                classifier.n_estimators))

    train_acc = mean_squared_error(y_train, pred_train)
    test_acc = mean_squared_error(y_test, pred_test)
    print("The {2} has a train mse: {0:.4f}, test mse: {1:.4f}".format(train_acc, test_acc,
                                                                       classifier.__class__.__name__))

    scores = cross_val_score(classifier, X, y, cv=10)
    print("Cross Validate Score: %0.3f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    # Calculate other evaluation metrics
    precision, recall, F1, _ = precision_recall_fscore_support(y_test, pred_test)
    print("precision: {0:.2f}. recall: {1:.2f}, F1: {2:.2f}".format(precision.mean(), recall.mean(), F1.mean()))

    classifier.fit(X_trans, y_train)
    print("Scaled X_train trained classifier has a accuracy of {0:4f}".format(
        accuracy_score(y_test, classifier.predict(X_test_trans))))
    # classifier.score(X_test_trans, y_test)

    print()
#