# coding: utf-8
"""
수치미분 vs 오차역전파법

속도 : 오차역 전파법이 더 빠름
복잡도 : 오차역 전파법이 더 복잡

따라서 비교적 단순한 수치미분을 사용해서 오차역전파법으로 구한 기울기를 검증함

(수치미분으로 계산한 기울기 - 오차역전파법으로 계산한 기울기 ) 오차가 0에 가까우면 잘 구현된 것.
"""
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 절대 오차의 평균을 구한다.
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))