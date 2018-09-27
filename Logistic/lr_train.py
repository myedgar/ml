#coding:UTF-8
import numpy as np
import os
from ml_math import sig

def error_rate(h, label):
    #计算当前的损失函数值
    m = np.shape(h)[0]

    sum_err = 0.0
    for i in xrange(m):
        if h[i, 0] > 0 and (1 - h[i, 0]) > 0:
            sum_err -= (label[i, 0] * np.log(h[i, 0]) + \
                    (1 - label[i, 0]) * np.log(1 - h[i, 0]))
        else:
            sum_err -= 0
    return sum_err / m

def lr_train_bgd(feature, label, max_cycle, alpha):
    #利用梯度下降法训练LR模型
    n = np.shape(feature)[1]      #特征个数
    w = np.mat(np.ones(n , 1))    #初始化权重
    i = 0
    while i <= max_cycle:
        i += 1
        h = sig(feature * w)
        err = label - h
        if i % 100 == 0:
            print("\t--------------iter=" + str(i) + \
            ", train error rate= " + str(error_rate(h, label)))
        w += (alpha * feature.T * err)
    return w

def load_data(file_name):
    f = open(file_name)
    
    feature_data = []
    label_data = []
    
    for line in f.readlines():
        feature_tmp = []
        label_tmp = []
        
        lines = line.strip().split("\t")
        feature_tmp.append(1)
        
        for i in xrange(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(float(lines[-1]))

        feature_data.append(feature_tmp)
        label_data.append(label_tmp)
    
    f.close()
    return np.mat(feature_data), np.mat(label_data)

def save_model(file_name, w):
    m = np.shape(w)[0]
    f_w = open(file_name, "w")

    w_array = []

    for i in xrange(m):
        w_array.append(str(w[i, 0]))
    f_w.write("\t".join(w_array))
    f_w.close()

if __name__ == "__main__":
    print("1.load data\n")
    feature, label = load_data("./Logistic/data.txt")

    print("2.训练\n")
    w = lr_train_bgd(feature, label, 1000, 0.01)

    print("3.保存\n")
    save_model("weight", w)