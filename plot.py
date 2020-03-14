import pandas as pd
import time
from itertools import cycle
from scipy import interp
from sklearn.externals.six import StringIO
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import pydotplus
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
def plot_roc(x_train, x_test, y_train, y_test):
    font1 = {'family': 'Arial',
             'weight': 'normal',
             'size': 10,
             }
    # Binarize the output
    y_train = label_binarize(y_train.tolist(), classes=['0', '1','2','3','4'])
    y_test = label_binarize(y_test.tolist(), classes=['0', '1','2','3','4'])
    n_classes = y_train.shape[1]
    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    x_train_samples, x_train_features = x_train.shape
    x_test_samples, x_test_features = x_test.shape
    x_train = np.c_[x_train, random_state.randn(x_train_samples, 200 * x_train_features)]
    x_test = np.c_[x_test, random_state.randn(x_test_samples, 200 * x_test_features)]

    # shuffle and split training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
    #                                                     random_state=0)

    # Learn to predict each class against the other
    classifier = DecisionTreeClassifier(criterion='entropy')
    y_score = classifier.fit(x_train, y_train).predict(x_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize='10',fontproperties='arial')
    plt.ylabel('True Positive Rate',fontsize='10',fontproperties='arial')
    plt.title('Receiver operating characteristic example',fontsize='10',fontproperties='arial')
    plt.legend(loc="lower right",prop=font1)
    #plt.show()
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize='10',fontproperties='arial')
    plt.ylabel('True Positive Rate',fontsize='10',fontproperties='arial')
    plt.title('Some extension of Receiver operating characteristic to tRNA,miRNA,siRNA',fontsize='10',fontproperties='arial')
    plt.legend(loc="lower right",prop=font1)
    plt.savefig('./output/roc_curve.png', format='png', dpi=120)
    #plt.show()
# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_name,
#                      class_names=target_name, filled=True, rounded=True,
#                      special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# #graph.write_pdf(r"C:\Users\dell\Desktop\deepnano-Alternatif\pdf\88Tree.pdf")
# print('Visible tree mi_plot saved as pdf.')