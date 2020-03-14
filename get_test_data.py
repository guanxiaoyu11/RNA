import argparse
import os
import h5py
# from helpers import *
import numpy as np
import matlab.engine
import csv
import argparse
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=r".\raw_data\test")
parser.add_argument('--plot_dir', type=str, default=r".\plot\test1")
parser.add_argument('--save_data_dir', type=str, default=r".\save_data\test")
parser.add_argument('--save_fea_dir', type=str, default=r".\save_fea\test")
args = parser.parse_args()
def get_test_data():
    eng = matlab.engine.start_matlab()
    # d=eng.abfload(r'C:\Users\dell\Desktop\trna\tRNA_4.abf','start',0,'stop','e')
    array1 = []
    array = [[]]
    file = args.data_dir + r"\mixture_0000.abf"
    d = eng.abfload(file, 'start', 0, 'stop', 'e')
    for i in range(len(d)):
        if d[i][0] < 250:
            array1.append(d[i][0])
        else:
            if i > 0:
                if d[i - 1][0] < 250:
                    #if array1 != [] and len(array1)>7000:
                    if (array1 != []) and (len(array1)>100) :
                        if np.min(np.array(array1))<-1000:
                            array1 = []
                        else:
                            array.append(array1)
                            array1 = []
                    else:
                        array1 = []
    del (array[0])
    eng.quit()
    return array
def plot():
    test_data = np.load(args.save_data_dir+r"\test1.npz")
    data = test_data['arr_0']
    array_c = []
    for i in range(len(data)):
        fft = np.fft.fft(data[i])
        fft_pow = np.abs(fft)
        data_mean = np.mean(data[i])
        data_std = np.std(data[i])
        data_var = np.var(data[i])
        data_max = np.max(data[i])
        data_min = np.min(data[i])
        data_med = np.percentile(data[i], 50)
        data_len = len(data[i])
        # a=data_1[i].describe()
        freqs = np.fft.fftfreq(len(fft))
        ###对数据进行平滑处理，4舍5入
        test = []
        for t in range(len(data[i])):
            if data[i][t] > 0:
                test.append(round(data[i][t], -1))
        counts = np.bincount(test)
        sort = np.sort(counts, -1)
        index1 = np.argwhere(counts == sort[len(sort) - 1])
        index2 = np.argwhere(counts == sort[len(sort) - 2])
        test1 = []
        for t in range(len(data[i])):
            if data[i][t] > 0:
                test1.append(round(data[i][t], 0))
        counts1 = np.bincount(test1)
        np.savez('test1.npz', counts1)
        sort1 = np.sort(counts1)[-10:]
        index3 = abs(np.argwhere(counts1 == sort1[9])[0][0] - np.argwhere(counts1 == sort1[0])[0][0])
        ###
        plt.plot(range(len(data[i])), data[i])
        file = args.plot_dir + r"\out_%i.png" % i
        plt.savefig(file)
        plt.close()
        #array_c.append([data_mean,data_std,data_var,data_max,data_min,data_med,fft_pow.max(),freqs.max(),data_len])
        array_c.append([data_mean, data_std, data_med,data_max, data_min, data_len,index1[0][0],index2[0][0],index3])
        print(data_mean, data_std, data_var, fft_pow.max(), freqs.max(), data_max, data_min)
    # del (array_c[0])
    save_fea_fir = args.save_fea_dir + r"\test_fea1.npz"
    np.savez(save_fea_fir, array_c)
#data=get_test_data()
#save_data=args.save_data_dir+r"\test1.npz"
#np.savez(save_data, data)
plot()

