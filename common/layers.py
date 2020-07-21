# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im
"""
>>>x = np.array([[]1.0,-0.5],[-2.0,3.0])
    >>>mask = (x <= 0)
    >>>print(mask)
    [[false,true],[true,false]]
"""
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x <= 0)
  

        out = x.copy()
        out[self.mask] = 0
        
        return out

    def backward(self,dout):
        #역전파 때는 순전파 때 만들어둔 mask를 써서 mask의 원소가 true인 곳에는 상류에서 전파된 dout을 0으로 설정
        dout[self.mask] = 0
        dx = dout

        return dx

class sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1/(1+np.exp(-x))
        self.out = out

        return out

    def backward(self,dout):
        dx = dout *(1.0 - self.out) *self.out

        return dx

#Affine 계층은 행렬의 형상을 일치 시켜주는 것 (도입부)
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx

#affine,ReLU 계층을 거쳐서 나온 점수(score)를 softmax 시켜서 정규화 시킴 (합이 1이 되게 - 확률)
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

#softmax 계층의 역전파는 (softmax 계층의 출력 값- 정답 레이블) 즉 , 오차 값을 앞으로 전해줌 
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx