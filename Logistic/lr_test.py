#coding:UTF-8
import numpy as np
from ml_math import sig

def load_weight(w):
    #导入LR模型
    f = open(w)
    w = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        w_tmp = []
        for x in lines:
            w_tmp.append(float(x))
        w.append(w_tmp)
    f.close()
    return np.mat(w)

def load_data(file_name, n):
    #导入测试数据
    f = open(file_name)
    feature_data = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split()
        if len(lines) <> n-1:
            continue
        feature_tmp.append(1)
        for x in lines:
            feature_tmp.append(float(x))
        feature_data.append(feature_tmp)
    f.close()
    return np.mat(feature_data)

def predict(data, w):
    #数据预测
    h = sig(data * w.T)
    m = np.shape(h)[0]
    for i in xrange(m):
        if h[i, 0] < 0.5:
            h[i, 0] = 0.0
        else:
            h[i, 0] = 1.0
    return h

def save_result(file_name, result):
    #保存最终的预测结果
    m = np.shape(result)[0]
    tmp = []
    for i in xrange(m):
        tmp.append(str(h[i, 0]))
    f_result = open(file_name, "w")
    f_result.write("\t".join(tmp))
    f_result.close()

if __name__ == "__main__":
    print("--------------1.load model--------------")
    w = load_weight("./Logistic/weight")
    n = np.shape(w)[1]

    print("--------------2.load data---------------")
    test_data = load_data("./Logistic/test_data", n)

    print("------------3.get prediction------------")
    h = predict(test_data, w)

    print("--------------4.save--------------------")
    save_result("result", h)