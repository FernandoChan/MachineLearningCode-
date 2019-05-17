import logging

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC, \
    AdaBoostClassifier as ABC, ExtraTreesClassifier

logger = logging.getLogger('RF Train')


def classification_dataset(name):
    data = arff.loadarff(name)
    df = pd.DataFrame(data[0])

    df = df.dropna()
    node_status_map = {b'B': 0, b'NB': 1, b"'P NB'": 2}
    df['Node Status'] = df['Node Status'].dropna().map(node_status_map).astype(int)  # encoding

    # class_map = {b"'NB-No Block'": 1, b"'No Block'": 2, b'Block': 0, b'NB-Wait': 3}
    # df['Class'] = df['Class'].dropna().map(class_map).astype(int)  # encoding

    X = df.drop(['Class', ], axis=1).values

    y = df.iloc[:, -1].values  # array

    X2 = df.iloc[:, :-3].values
    return X, y


X, y = classification_dataset('OBS-Network-DataSet_2_Aug27.arff')

from sklearn.preprocessing import OneHotEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

one_hot = OneHotEncoder()
label_encoder = LabelEncoder()
label_encoder.fit(y)
encode_y = label_encoder.transform(y)
dummy_y = np_utils.to_categorical(encode_y)

# print( y[:10])

# # Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rfc_clf = RFC(n_estimators=7, criterion='gini', random_state=42, oob_score=False,
              max_features="auto", )

gbc_clf = GBC(random_state=42, )
abc_clf = ABC(random_state=42, )
extra_tree = ExtraTreesClassifier(n_estimators=10, random_state=42)
clf = [rfc_clf, gbc_clf, abc_clf, extra_tree]

params = {"n_estimators": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          # "criterion":["entropy", "gini"],
          # "n_jobs":[1,-1],
          # "warm_start":[True,False],
          # "max_depth" : list(range (10,100,10))
          }
#
# grid = GridSearchCV(cv=10, estimator=rfc_clf, param_grid=params)
# grid.fit(X_train, y_train)
# print("grid score is {0}".format(grid.best_score_))
# print("The grid best param is {0}".format(grid.best_params_))
#
# best_forest = grid.best_estimator_
# clf.append(best_forest)
# sorted(grid.cv_results_.keys())
#
# metric = [accuracy_score, mean_squared_error]
# scaler = preprocessing.StandardScaler.fit(self=preprocessing.StandardScaler(), X=X_train)
# X_trans = scaler.transform(X_train)
# X_test_trans = scaler.transform(X_test)
#
# for classifier in clf:
#     classifier.fit(X_train, y_train)
#     pred_train = classifier.predict(X_train)
#     pred_test = classifier.predict(X_test)
#     train_acc = accuracy_score(y_train, pred_train)
#     test_acc = accuracy_score(y_test, pred_test)
#     print("The {2} has {3} estimators , got a train score: {0:.4f}, test score: {1:.4f}".format(train_acc, test_acc,
#                                                                                                 classifier.__class__.__name__,
#                                                                                                 classifier.n_estimators))
#
#     train_acc = mean_squared_error(y_train, pred_train)
#     test_acc = mean_squared_error(y_test, pred_test)
#     print("The {2} has a train mse: {0:.4f}, test mse: {1:.4f}".format(train_acc, test_acc,
#                                                                        classifier.__class__.__name__))
#
#     scores = cross_val_score(classifier, X, y, cv=10)
#     print("Cross Validate Score: %0.3f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
#
#     # Calculate other evaluation metrics
#     precision, recall, F1, _ = precision_recall_fscore_support(y_test, pred_test)
#     print("precision: {0:.2f}. recall: {1:.2f}, F1: {2:.2f}".format(precision.mean(), recall.mean(), F1.mean()))
#
#     classifier.fit(X_trans, y_train)
#     print("Scaled X_train trained classifier has a accuracy of {0:4f}".format(
#         accuracy_score(y_test, classifier.predict(X_test_trans))))
#     # classifier.score(X_test_trans, y_test)
#
#     print()
# #
logger.log(level=1, msg="start to fit grid")

# Evaluation
# 评价方法：大一点的数据集采用10-折交叉验证（10-fold cross validation），小一点的数据集（如200以下）采用留一测试。


seed = 42
np.random.seed(seed)

from keras.models import Sequential
from keras.layers import Dense


# def cnn_pipeline(inputs):
#     model = Sequential()
#     model.add(Dense(8, input_dim=))
#     model.add(Conv2D(32, kernel_size=(3, 3),
#                      activation='relu',
#                      input_shape=input_shape))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes, activation='softmax'))
#
#     model.compile(loss=keras.losses.categorical_crossentropy,
#                   optimizer=keras.optimizers.Adadelta(),
#                   metrics=['accuracy'])
#
#     model.fit(X_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               verbose=1,
#               validation_data=(x_test, y_test))
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
estimator = KerasClassifier(build_fn=baseline_model(), epochs=200, batch_size=5, verbose=0)

# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

model = baseline_model()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
