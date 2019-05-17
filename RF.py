import  numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC , GradientBoostingClassifier as GBC, AdaBoostClassifier as ABC
from scipy.io import arff
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support , mean_squared_error
from sklearn import preprocessing
import  logging
logger = logging.getLogger('RF Train')
def classification_dataset(name):
    data = arff.loadarff(name)
    df = pd.DataFrame(data[0])

    df = df.dropna()
    node_status_map = {b'B': 0, b'NB': 1 , b"'P NB'":2}
    df['Node Status'] = df['Node Status'].dropna().map(node_status_map).astype(int) # encoding

    class_map = {b"'NB-No Block'": 1, b"'No Block'": 2, b'Block': 0, b'NB-Wait': 3}
    df['Class'] = df['Class'].dropna().map(class_map).astype(int) # encoding

    X = df.drop(['Class',], axis=1).values

    y = df.iloc[:, -1].values # array

    X2 = df.iloc[:, :-3].values
    return X,y
X,y =  classification_dataset('OBS-Network-DataSet_2_Aug27.arff')
# print( y[:10])

# # Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

from sklearn.model_selection import train_test_split, GridSearchCV , KFold ,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)




rfc_clf = RFC(n_estimators=100, criterion='gini', random_state=42 , oob_score=False,
          max_features="auto" ,)
gbc_clf = GBC(random_state=42 , )
abc_clf = ABC(random_state=42,)
clf = [rfc_clf,gbc_clf,abc_clf]

params = {"n_estimators":[10,20,50,100,150] ,
          "criterion":["entropy", "gini"],
          "n_jobs":[1,-1],
          "warm_start":[True,False],
          "max_depth" : list(range (10,100,10))
        }

metric = [accuracy_score, mean_squared_error]
scaler = preprocessing.StandardScaler.fit(self=preprocessing.StandardScaler(), X = X_train)
X_trans = scaler.transform(X_train)
X_test_trans = scaler.transform(X_test)
for classifier in clf:
    classifier.fit(X_train, y_train)
    pred_train = classifier.predict(X_train)
    pred_test = classifier.predict(X_test)
    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    print("The {2} has a train score: {0:.4f}, test score: {1:.4f}".format(train_acc, test_acc,classifier.__class__.__name__))

    train_acc = mean_squared_error(y_train, pred_train)
    test_acc = mean_squared_error(y_test, pred_test)
    print("The {2} has a train mse: {0:.4f}, test mse: {1:.4f}".format(train_acc, test_acc,classifier.__class__.__name__))

    scores = cross_val_score(classifier, X , y , cv= 10 )
    print("Cross Validate Score: %0.3f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    classifier.fit(X_trans, y_train)
    print("Scaled X_train trained classifier has a accuracy of {0:4f}".format(accuracy_score(y_test, classifier.predict(X_test_trans))))
    # classifier.score(X_test_trans, y_test)

    print()
# from sklearn.utils.multiclass import type_of_target
#
logger.log(level=1, msg= "start to fit grid")

# grid = GridSearchCV(cv = 10, estimator=rfc_clf , param_grid=params)
# grid.fit(X_train, y_train)
# sorted(grid.cv_results_.keys())


# Evaluation
# 评价方法：大一点的数据集采用10-折交叉验证（10-fold cross validation），小一点的数据集（如200以下）采用留一测试。




