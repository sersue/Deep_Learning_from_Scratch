# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

"""
TwoLayerNet클래스가 사용하는 변수
-----------------------
params : 신경망의 매개변수를 보관하는 딕셔너리 변수
layers : 순서가 있는 딕셔너리변수, 신경망의 계층을 보관 (ex. layers['Affine1],lagers['Relu1'])
lastLayer : 신경망의 마지막 계층 , 여기선 SoftmaxWithLoss 계층

TwoLayerNet 클래스의 메서드
-----------------------
__init__(self, input_size, hidden_size, output_size)
predict(selt,x)
loss(selt,x,t)
accuracy(selt,x,t)
numerical_gradient(selt,x,t) :가중치의 매개변수의 기울기를 수치 미분 방식으로 구한다.
gradient : 가중치의 매개변수의 기울기를 오차역전파법으로 구한다.
"""

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성(OrderedDict : 순서가 있는 딕셔너리, 추가한 순서대로 각 계층의 forward() 메서드를 호출하면 됨.)
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    #오차역전파법 
    def gradient(self, x, t):
        # forward 
        self.loss(x, t)

        # backward (계층을 반대 순서로 호출하면 됨.) 
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())

        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads