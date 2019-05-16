import  numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from scipy.io import arff
import pandas as pd
import os
def classification_dataset(name):
    data = arff.loadarff(name)
    df = pd.DataFrame(data[0])

    df = df.dropna()
    node_status_map = {b'B': 0, b'NB': 1 , b"'P NB'":2}
    df['Node Status'] = df['Node Status'].dropna().map(node_status_map).astype(int) # encoding

    class_map = {b"'NB-No Block'": 1, b"'No Block'": 2, b'Block': 0, b'NB-Wait': 3}
    df['Class'] = df['Class'].dropna().map(class_map).astype(int) # encoding

    X = df.drop(['Class',], axis=1)
    y = df.iloc[:, -1].values # array

    X2 = df.iloc[:, :-3].values
    return X,y
X,y =  classification_dataset('OBS-Network-DataSet_2_Aug27.arff')
print(X[:10], y[:10])

# # Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

clf = RFC(n_estimators=10, criterion='entropy', random_state=43 , oob_score=True,
          max_features="auto" ,)
params = {"n_estimators":[10,20,50,100,150] ,
          "criterion":["entropy", "gini"],
          "n_jobs":[1,-1],
          "warm_start":[True,False],
          "max_depth" : list(range (10,100,10))
        }
clf.fit(X_train, y_train)
# todo Debug , 为啥出错
# from sklearn.utils.multiclass import type_of_target

# grid = GridSearchCV(cv = 10, estimator=clf , param_grid=params)
# grid.fit(X_train, y_train)
# sorted(grid.cv_results_.keys())

# pred_train = grid.predict(X_train)
# pred_test = grid.predict(X_test)
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# train_acc = accuracy_score(y_train, pred_train)
# test_acc = accuracy_score(y_test, pred_test)
# print ("train acc: {0:.2f}, test acc: {1:.2f}".format(train_acc, test_acc))
# Evaluation
# 评价方法：大一点的数据集采用10-折交叉验证（10-fold cross validation），小一点的数据集（如200以下）采用留一测试。




