# encoding=utf-8

import pandas as pd
import time
from itertools import cycle
from scipy import interp
from sklearn.externals.six import StringIO
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, precision_score, \
    recall_score, f1_score, classification_report
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import pydotplus
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import accuracy_score
import argparse
import plot
import os
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from IPython.display import Image
parser = argparse.ArgumentParser()
parser.add_argument('--label_dir', type=str, default=r".\label")
parser.add_argument('--fea_dir', type=str, default=r".\save_fea")
parser.add_argument('--output_tree_dir', type=str, default=r".\pdf")
parser.add_argument('--label_dir1', type=str, default=r".\label\label1")
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
    file = args.label_dir + r"\test_label2.txt"
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
    x=[[]]
    for i in range(len(x_t)):
        if i in ta_label_list:
            x.append(x_t[i])
    for t in range(len(x_t)):
        if t in tb_label_list:
            x.append(x_t[t])
    for s in range(len(x_m)):
        if s in mi_label_list:
            x.append(x_m[s])
    for n in range(len(x_s)):
        if n in si_label_list:
            x.append(x_s[n])
    #x = np.append(x_t, x_m[0:100], axis=0)
    #x= np.append(x, x_s[0:289], axis=0)
    del (x[0])
    x=np.array(x)
    return x

def get_train_label():
    array = [[]]
    file = args.label_dir + r"\train_label1.txt"
    f = open(file)
    lines = f.readlines()
    for i in range(len(lines)):
        list = lines[i].split()
        #del (list[0])
        array.append(list)
    del (array[0])
    #array=np.array(array)
    ta_label_list, tb_label_list, mi_label_list, si_label_list = get_file_name()
    array1 = [[]]
    array1.append(list)
    for i in range(len(ta_label_list)):
        array1.append(['1'])
    for i in range(len(tb_label_list)):
        array1.append(['2'])
    for i in range(len(mi_label_list)):
        array1.append(['3'])
    for i in range(len(si_label_list)):
        array1.append(['4'])
    del (array1[0])
    del (array1[1])
    array1 = np.array(array1)
    return array1
def get_test_data1():
    test_file = args.fea_dir + r"\test\test_fea1.npz"
    x_t = np.load(test_file)['arr_0']
    return x_t[0:586]
def get_test_label1():
    array = [[]]
    file = args.label_dir1 + r"\test_label1.txt"
    f = open(file)
    lines = f.readlines()
    for i in range(len(lines)):
        list = lines[i].split()
        #del (list[0])
        array.append(list)
    del (array[0])
    array=np.array(array)
    return array
def get_train_data1():
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

def get_train_label1():
    array = [[]]
    file = args.label_dir1 + r"\1.txt"
    f = open(file)
    lines = f.readlines()
    for i in range(len(lines)):
        list = lines[i].split()
        #del (list[0])
        array.append(list)
    del (array[0])
    array=np.array(array)
    return array
def plot_learning_curve(algo, X_train, X_test, y_train, y_test):
    """绘制学习曲线：只需要传入算法(或实例对象)、X_train、X_test、y_train、y_test"""
    """当使用该函数时传入算法，该算法的变量要进行实例化，如：PolynomialRegression(degree=2)，变量 degree 要进行实例化"""
    train_score = []
    test_score = []
    for i in range(1, len(X_train) + 1):
        algo.fit(X_train[:i], y_train[:i])

        y_train_predict = algo.predict(X_train[:i])
        train_score.append(accuracy_score(y_train[:i], y_train_predict))

        y_test_predict = algo.predict(X_test)
        test_score.append(accuracy_score(y_test, y_test_predict))

    # plt.plot([i for i in range(1, len(X_train) + 1)],
    #          np.sqrt(train_score), label="train")
    plt.plot([i for i in range(1, len(X_train) + 1)],
             test_score, label="test")

    plt.legend()
    plt.axis([0, len(X_train) + 1, 0, 1])
    plt.title("learning  curve of classifar ")
    #plt.show()


def cm_plot(y, yp):
    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数

    cm = confusion_matrix(y, yp)  # 混淆矩阵

    import matplotlib.pyplot as plt  # 导入作图库
    plt.matshow(cm, cmap=plt.cm.Greens)  # 画混淆矩阵图，配色风格使用cm.Greens，更多风格请参考官网。
    plt.colorbar()  # 颜色标签

    for x in range(len(cm)):  # 数据标签
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    return plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




if __name__ == '__main__':

    print("Start read raw_data...")
    time_1 = time.time()
    x_train1 = get_train_data1()
    y_train1 = get_train_label1()
    x_test1 = get_test_data1()
    y_test1 = get_test_label1()
    x_train=get_train_data()
    y_train=get_train_label()
    x_test=get_test_data()
    y_test=get_test_label()
    print(x_train.shape, y_train.shape,x_test.shape,y_test.shape)
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

    time_2 = time.time()
    print('read raw_data cost %f seconds' % (time_2 - time_1))


    print('Start training...')
    # criterion可选‘gini’, ‘entropy’，默认为gini(对应CART算法)，entropy为信息增益（对应ID3算法）mse
    from sklearn.metrics import mean_squared_error

    # 存储每一次训练的模型的均方误差
    train_score = []
    test_score = []
    best_score=0
    # for 循环：进行 75 次模型训练，每次训练出 1 个模型，第一次给 1 个数据，第二次给 2 个数据，... ，最后一次给 75 个数据
    for i in range(1, args.epochs):
        clf = DecisionTreeClassifier(criterion='entropy')
        clf1 = DecisionTreeClassifier(criterion='entropy')
        clf.fit(x_train, y_train)
        clf1.fit(x_train1, y_train1)
        time_3 = time.time()
        print('training cost %f seconds' % (time_3 - time_2))
        y_train_predict = clf.predict(x_train)
        y_train_predict1 = clf1.predict(x_train1)
        train_score.append(accuracy_score(y_train, y_train_predict))
        print('Start predicting...')

        time_4 = time.time()

        y_test_predict = clf.predict(x_test)
        y_test_predict1 = clf1.predict(x_test1)
        for s in range(len(y_test_predict)):
            if y_test_predict1[s]=='0':
                y_test_predict[s]='0'
        print('predicting cost %f seconds' % (time_4 - time_3))
        test_score.append(accuracy_score(y_test, y_test_predict))
        score = accuracy_score(y_test, y_test_predict)
        print("The accruacy score is %f" % score)
        #best_score=score
        if score >best_score:
            best_score=score
        print("the best accruacy score is %f" % best_score)
    #学习曲线epoch精度的变化
    # plt.plot([i for i in range(1, args.epochs)], train_score, label='train')
    #plt.plot([i for i in range(1, args.epochs)], test_score, label='test')
    #plt.legend()
    #plt.title('Test Accuracy curve for each epoch')
    #plt.show()

    #学习曲线，测试样本变化对精度的变化
    #plot_learning_curve(DecisionTreeClassifier(criterion='entropy'), x_train1, x_test1, y_train1, y_test1)
    #plot_learning_curve(DecisionTreeClassifier(criterion='entropy'),x_train, x_test, y_train, y_test)

    for s in range(len(y_test)):
        if y_test[s]!=y_test_predict[s] :
            print(y_test[s],y_test_predict[s],s)

    #data_mean,data_std,data_var,data_max,data_min,data_med,fft_pow.max(),freqs.max(),data_len
    #feature_name = ['data_mean','data_std','data_var','data_max','data_min','data_med','fft_pow_max','freqs.max','data_len']
    feature_name = ['data_mean', 'data_std', 'data_med', 'data_max', 'data_min','data_len','step_1','step_2','noise']
    target_name1 = ['0', '1']
    target_name = ['1', '2', '3', '4']
    dot_data=tree.export_graphviz(clf, out_file=None,
                         feature_names=feature_name,
                         class_names=target_name,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())
    graph.write_pdf(args.output_tree_dir+'/tree1.pdf')
    # 特征重要性
    y_importances = clf.feature_importances_
    y_importances1 = clf1.feature_importances_
    x_importances = feature_name
    y_pos = np.arange(len(x_importances))
    # 横向柱状图
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23,
             }
    plt.barh(y_pos, y_importances, align='center')
    plt.yticks(y_pos, x_importances,fontsize='10',fontproperties='arial')
    plt.xlabel('Importances',fontsize='10',fontproperties='arial')
    plt.xlim(0, 1)
    plt.title('Features Importances',fontsize='10',fontproperties='arial')
    plt.barh(y_pos, y_importances1, align='center')
    plt.yticks(y_pos, x_importances,fontsize='10',fontproperties='arial')
    plt.xlabel('Importances',fontsize='10',fontproperties='arial')
    plt.xlim(0, 1)


    plt.legend(['classifier1', 'classifier2'])
    plt.title('Features Importances',fontsize='10',fontproperties='arial')
    plt.savefig('./output/features_importances.png', format='png',dpi=120)
    # #绘制数据分布散点图（附件中可以用）
    # plt.show()
    # plt.scatter(x_train[:,0].tolist(), y_train.tolist())
    # plt.show()
    # plt.scatter(x_train[:, 1].tolist(), y_train.tolist())
    # plt.show()
    # plt.scatter(x_train[:, 2].tolist(), y_train.tolist())
    # plt.show()
    # plt.scatter(x_train[:, 3].tolist(), y_train.tolist())
    # plt.show()
    # plt.scatter(x_train[:, 4].tolist(), y_train.tolist())
    # plt.show()
    # plt.scatter(x_train[:, 5].tolist(), y_train.tolist())
    # plt.show()
    precision, recall, F1, _ = precision_recall_fscore_support(y_test, y_test_predict)
    print(precision, recall, F1)
    plot.plot_roc(x_train, x_test, y_train, y_test)
    #绘制混淆矩阵
    obj1 = confusion_matrix(y_test, y_test_predict)
    print('confusion_matrix\n', obj1)
    classes = list(set(y_test_predict))
    classes.sort()
    plt.figure(figsize=(12, 8), dpi=100)
    plt.imshow(obj1, cmap=plt.cm.Blues)

    indices = range(len(obj1))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('label of test set sample')
    plt.ylabel('label of predict')
    plt.title('Confusion Matrix',fontsize='10',fontproperties='arial')
    for first_index in range(len(obj1)):
        for second_index in range(len(obj1[first_index])):
            plt.text(first_index, second_index, obj1[first_index][second_index],fontsize=10, va='center', ha='center')
            #plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')

    #plt.show()
    labels = ['0', '1', '2', '3', '4']
    cm = confusion_matrix(y_test, y_test_predict)
    tick_marks = np.array(range(len(labels))) + 0.5

    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        cm = confusion_matrix(y_test, y_test_predict)
        np.set_printoptions(precision=2)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # show confusion matrix
    plt.savefig('./output/confusion_matrix.png', format='png')
    plt.show()

    #绘制分类报告
    r = classification_report(y_test, y_test_predict)
    print('分类报告为：', r, sep='\n')
    # 绘制分类边界线
    l, r = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
    b, t = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
    n = 500
    grid_x, grid_y = np.meshgrid(np.linspace(l, r, n), np.linspace(b, t, n))
    bg_x = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    bg_y = clf.predict(bg_x)
    grid_z = bg_y.reshape(grid_x.shape)

    # 画图
    plt.figure('NB Classification', facecolor='lightgray')
    plt.title('NB Classification', fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.tick_params(labelsize=10)
    plt.pcolormesh(grid_x, grid_y, grid_z, cmap='gray')
    plt.scatter(x_test[:, 0], x_test[:, 1], s=80, c=y_test, cmap='jet', label='Samples')

    plt.legend()
    #plt.show()

    print('accuracy:{}'.format(accuracy_score(y_test, y_test_predict)))
    print('precision:{}'.format(precision_score(y_test, y_test_predict, average='micro')))
    print('recall:{}'.format(recall_score(y_test, y_test_predict, average='micro')))
    print('f1-score:{}'.format(f1_score(y_test, y_test_predict, average='micro')))
# plot_mushroom_boundary(x_test, y_test, clf)
