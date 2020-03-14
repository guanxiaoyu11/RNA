# encoding=utf-8

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

if __name__ == '__main__':

    print("Start read raw_data...")
    time_1 = time.time()

    x = np.load("data_fea_1.npz")['arr_0']
    y = np.load("data_label_1.npz")['arr_0']
    for s in range(2, 5):
        data_fea = r"data_fea_%s.npz" % s
        data_label = r"data_label_%s.npz" % s
        x = np.append(x, np.load(data_fea)['arr_0'], axis=0)
        y = np.append(y, np.load(data_label)['arr_0'], axis=0)
    print(x.shape, y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    time_2 = time.time()
    print('read raw_data cost %f seconds' % (time_2 - time_1))


    print('Start training...')
    # n_estimators表示要组合的弱分类器个数；
    # algorithm可选{‘SAMME’, ‘SAMME.R’}，默认为‘SAMME.R’，表示使用的是real boosting算法，‘SAMME’表示使用的是discrete boosting算法
    clf = AdaBoostClassifier(n_estimators=100,algorithm='SAMME.R')
    clf.fit(x_train,y_train)
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))


    print('Start predicting...')
    test_predict = clf.predict(x_test)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))


    score = accuracy_score(y_test, test_predict)
print("The accruacy score is %f" % score)