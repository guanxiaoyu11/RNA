# encoding=utf-8

import time
from itertools import cycle
from scipy import interp
from sklearn.externals.six import StringIO
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--label_dir', type=str, default=r"C:\Users\dell\Desktop\deepnano-Alternatif\label\label1")
parser.add_argument('--fea_dir', type=str, default=r"C:\Users\dell\Desktop\deepnano-Alternatif\save_fea")
parser.add_argument('--output_tree_dir', type=str, default=r"C:\Users\dell\Desktop\deepnano-Alternatif\pdf")
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()

def plot_mushroom_boundary(X, y, fitted_model):
    plt.figure(figsize=(9.8, 5), dpi=100)

    for i, plot_type in enumerate(['Decision Boundary', 'Decision Probabilities']):
        plt.subplot(1, 2, i + 1)

        mesh_step_size = 100  # step size in the mesh
        x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
        y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))
        if i == 0:
            Z = fitted_model.predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            try:
                Z = fitted_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            except:
                plt.text(0.4, 0.5, 'Probabilities Unavailable', horizontalalignment='center',
                         verticalalignment='center', transform=plt.gca().transAxes, fontsize=12)
                plt.axis('off')
                break
        Z = Z.reshape(xx.shape)
        plt.scatter(X[y.values == 0, 0], X[y.values == 0, 1], alpha=0.4, label='Edible', s=5)
        plt.scatter(X[y.values == 1, 0], X[y.values == 1, 1], alpha=0.4, label='Posionous', s=5)
        plt.imshow(Z, interpolation='nearest', cmap='RdYlBu_r', alpha=0.15,
                   extent=(x_min, x_max, y_min, y_max), origin='lower')
        plt.title(plot_type + '\n' +
                  str(fitted_model).split('(')[0] + ' Test Accuracy: ' + str(np.round(fitted_model.score(X, y), 5)))
        plt.gca().set_aspect('equal');

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.08, wspace=0.02)
def get_test_data():
    test_file = args.fea_dir + r"\test\test_fea1.npz"
    x_t = np.load(test_file)['arr_0']
    return x_t[0:586]
def get_test_label():
    array = [[]]
    file = args.label_dir + r"\test_label1.txt"
    f = open(file)
    lines = f.readlines()
    for i in range(len(lines)):
        list = lines[i].split()
        #del (list[0])
        array.append(list)
    del (array[0])
    array=np.array(array)
    return array
def get_file_name():
    ta_label_plot = args.label_dir + r"\plot\t_plot\typea"
    tb_label_plot = args.label_dir + r"\plot\t_plot\typeb"
    m_label_plot = args.label_dir + r"\plot\miplot"
    s_label_plot = args.label_dir + r"\plot\siplot"
    ta_label_list = []
    tb_label_list = []
    mi_label_list = []
    si_label_list = []
    for root, dirs, files in os.walk(ta_label_plot):
        for file in files:
            ta_label_list.append(int(file.split('_')[1].split('.')[0]))
    for root, dirs, files in os.walk(tb_label_plot):
        for file in files:
            tb_label_list.append(int(file.split('_')[1].split('.')[0]))
    for root, dirs, files in os.walk(m_label_plot):
        for file in files:
            mi_label_list.append(int(file.split('_')[1].split('.')[0]))
    for root, dirs, files in os.walk(s_label_plot):
        for file in files:
            si_label_list.append(int(file.split('_')[1].split('.')[0]))
    for i in range(1000):
        if i in (tb_label_list) and (i in ta_label_list):
            print(i)
    return ta_label_list,tb_label_list,mi_label_list,si_label_list
def get_train_data():
    array = [[]]
    t_file = args.fea_dir + r"\tRNA\tRNA_fea.npz"
    m_file = args.fea_dir + r"\miRNA\miRNA_fea.npz"
    s_file = args.fea_dir + r"\siRNA\siRNA_fea.npz"

    ta_label_list,tb_label_list,mi_label_list,si_label_list=get_file_name()
    x_t = np.load(t_file)['arr_0']
    x_m = np.load(m_file)['arr_0']
    x_s = np.load(s_file)['arr_0']

    x = np.append(x_t[0:340], x_m[0:500], axis=0)
    x= np.append(x, x_s[0:500], axis=0)
    x=np.array(x)
    return x

def get_train_label():
    array = [[]]
    file = args.label_dir + r"\1.txt"
    f = open(file)
    lines = f.readlines()
    for i in range(len(lines)):
        list = lines[i].split()
        #del (list[0])
        array.append(list)
    del (array[0])
    array=np.array(array)
    return array

if __name__ == '__main__':

    print('prepare datasets...')
    x_train = get_train_data()
    y_train = get_train_label()
    x_test = get_test_data()
    y_test = get_test_label()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    time_2=time.time()
    print('Start training...')
    clf = svm.SVC()  # svm class
    clf.fit(x_train, y_train)  # training the svc model
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start predicting...')
    test_predict=clf.predict(x_test)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(y_test, test_predict)
print("The accruacy score is %f" % score)
