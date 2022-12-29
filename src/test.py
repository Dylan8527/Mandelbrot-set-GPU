# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:01:29 2019
project name:Mandelbrot set
@author: 帅帅de三叔
"""
import numpy as np
import matplotlib.pyplot as plt

def iterator(c,r,max_iter):#定义逃逸时间函数，c为初始值，r为收敛半径，max_iter为最大迭代次数，返回逃逸时间
    z=c #初始值
    for iter in range(0,max_iter,1): 
        if abs(z)>r:break
        z=z**2+c 
    return iter 

def plot_mandelbrot(): #定义绘制mandelbrot图像 
    X=np.linspace(-1.75,1.05,5000) #实部范围，5000这个数要量力而行
    Y=np.linspace(-1.25,1.25,5000) #虚部范围，5000这个数要量力而行
    real,image=np.meshgrid(X, Y) #生成网格点坐标矩阵。
    c=real+image*1j #构造复数
    mandelbrot_set = np.frompyfunc(iterator, 3, 1)(c, 1.5, 100).astype(np.float) #frompyfunc(func, nin, nout)，其中func是需要转换的函数，nin是函数的输入参数的个数，nout是此函数的返回值的个数,frompyfunc把Python里的函数（可以是自写的）转化成ufunc
    plt.figure(dpi=600) #dpi设置分辨率尽可能高，细节显示更炫
    plt.imshow(mandelbrot_set,extent=[-1.35, 1.35, -1.25, 1.25]) #extent用来调节显示框大小比例
    #plt.axis('off') #是否显示坐标轴
    plt.show()

if __name__=="__main__":
    plot_mandelbrot()
