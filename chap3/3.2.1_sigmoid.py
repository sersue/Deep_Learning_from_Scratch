# -*- coding: utf-8 -*-
#sigmoid는 s자 모양 함수 : 신경망에서 자주 이용하는 활성화 함수(압력신호의 총합이 활성화를 일으키는지 정하는 역할)
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0,5.0,0.1)
y = sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()