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
parser.add_argument('--data_dir', type=str, default=r".\raw_data\trna")
parser.add_argument('--plot_dir', type=str, default=r".\plot\t_plot")
parser.add_argument('--save_data_dir', type=str, default=r".\save_data\trna")
parser.add_argument('--save_fea_dir', type=str, default=r".\save_fea\trna")
args = parser.parse_args()
def get_train_data():
    eng = matlab.engine.start_matlab()
    # d=eng.abfload(r'C:\Users\dell\Desktop\trna\tRNA_4.abf','start',0,'stop','e')
    array1 = []
    array = [[]]
    for s in range(1,5):
        file = args.data_dir + r"\tRNA_%s.abf" % s
        d = eng.abfload(file, 'start', 0, 'stop', 'e')
        for i in range(len(d)):
            if d[i][0] < 250 :
                array1.append(d[i][0])
            else:
                if i > 0:
                    if d[i - 1][0] < 250:
                        # if array1 != [] and len(array1)>7000:
                        if array1 != [] and len(array1) >10000:
                            array.append(array1)
                        array1 = []
    del (array[0])
    eng.quit()
    return array
def plot():
    trna_data = np.load(args.save_data_dir + r"\tRNA.npz")
    data = trna_data['arr_0']
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
        # a=data_1[i].describe()
        data_len = len(data[i])
        freqs = np.fft.fftfreq(len(fft))
        # plt.plot(range(len(data[i])), data[i])
        # file = args.plot_dir + r"\out_%i.png" % i
        # plt.savefig(file)
        # plt.close()
        #求数据变化率
        # r = []
        # data_arr = data[i]
        # for t in range(len(data_arr) - 1):
        #     r.append(float(data_arr[t] - data_arr[t + 1]) / float(data_arr[t]))
        ####
        ###fft滤波
        # test_y = fft
        # for t in range(len(fft)):
        #     if t <= (len(fft) - 100):
        #         test_y[t] = 0
        # test = np.fft.ifft(test_y)  # 对变换后的结果应用ifft函数，应该可以近似地还原初始信号。

        # test = pd.Series(test)
        # # test_diff1 = DataFrame.diff(a=test,n=1)
        # test_diff1 = test.diff(4)
        # test_diff1.dropna()
        # test_diff1 = test_diff1[2000:]
        # test_diff1 = test_diff1.values
        ###
        ###对数据进行平滑处理，4舍5入
        test = []
        for t in range(len(data[i])):
            if data[i][t] > 0:
                test.append(round(data[i][t], -1))
        counts = np.bincount(test)
        sort=np.sort(counts,-1)
        index1=np.argwhere(counts==sort[len(sort)-1])
        index2 = np.argwhere(counts == sort[len(sort)-2])
        test1 = []
        for t in range(len(data[i])):
            if data[i][t] > 0:
                test1.append(round(data[i][t], 0))
        counts1 = np.bincount(test1)
        np.savez('trna.npz', counts1)
        sort1 = np.sort(counts1)[-10:]
        index3 = abs(np.argwhere(counts1 == sort1[9])[0][0]-np.argwhere(counts1 == sort1[0])[0][0])
        ###
        plt.plot(range(len(data[i])), data[i])
        file = args.plot_dir + r"\out_%i.png" % i
        plt.savefig(file)
        plt.close()
        # array_c.append([data_mean,data_std,data_var,data_max,data_min,data_med,fft_pow.max(),freqs.max(),data_len])
        array_c.append([data_mean, data_std, data_med,data_max, data_min, data_len,index1[0][0],index2[0][0],index3])
        #print(data_mean, data_std, data_var, fft_pow.max(), freqs.max(), data_max, data_min)
        print(index1,index2,index3)
    # del (array_c[0])
    save_fea_fir = args.save_fea_dir + r"\tRNA_fea.npz"
    np.savez(save_fea_fir, array_c)
#data=get_train_data()
#save_data=args.save_data_dir+r"\tRNA.npz"
#np.savez(save_data, data)
plot()

