#coding:UTF-8
import numpy as np

def sig(x):
    #Sigmid函数
    return 1.0 / (1 + np.exp(-x))