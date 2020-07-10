# -*- coding: utf-8 -*-
""" 퍼셉트론은 가중치를 사람이 직접 정해줘야한다는 단점이 있음 
하지만 신경망은 데이터로부터 자동으로 학습함 """
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0,dtype = np.int)

x = np.arange(-5.0,5.0,0.1)
y = step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()