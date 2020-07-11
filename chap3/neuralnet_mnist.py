# -*- coding: utf-8 -*-
import sys,os
import pickle
sys.path.append(os.pardir)
import numpy as np 
from dataset.mnist import load_mnist
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def get_data():
    (x_train,t_train),(x_test,t_test) = \
        load_mnist(flatten=True, normalize=True , one_hot_label=False)
    """
    normalize를 True로 설정하면 0.0~1.0 범위로 변환된다
    이처럼 데이터를 특정 범위로 변환 처리 하는 것을 '정규화'(normalization)라고 하고 
    변환을 가하는 것을 '전처리'(pre-processing) 라고 한다
    ---------------------
    '전처리' 작업으로 '정규화'를 수행한 것

    """
    return x_test,t_test

#pickle 파일인 sample_weight.pkl에 저장된 '학습된 가중치 매개변수'를 읽음
def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)

    return network

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def predict(network ,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) +b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) +b3
    y = softmax(a3)

    return y
# 방법 1 (한장의 사진씩 넣는 것)
x,t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network,x[i])
    """
    predict()
    ------------
    각 레이블의 확률을 넘파이 배열로 반환
    예를 들어 [0.1,0.3,0.2 ... 0.04]
    같은 배열이 반환되며,
    이는 0일 확률이 0.1, 1일 확률이 0.3 임을 의미 
    """
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

"""방법 2 - batch (묶음으로 100장씩 넣는 것) 
    이렇게 하면 I/O 입력으로 데이터를 읽는 횟수가 줄어 
    배치 처리를 수행함으로써 큰 배열을 한꺼번에 계산
    한꺼번에 계산 하는 것이 분할된 작은 배열을 여러번 계산하는 것보다 빠름
"""


x,t = get_data()
network = init_network()

batch_size =100
accuracy_cnt = 0

for i in range(0,len(x),batch_size):

    x_batch = x[i:i+batch_size]
    y_batch= predict(network,x_batch)
    p = np.argmax(y_batch, axis=1) 
    """ axis =1
     100*10 배열 중 최댓값의 인덱스를 찾도록 하는 것
     예를 들어 x = np.array([[0.1,0.8,0.1],[0.3,0.1,0.6]]) 이면
     y = np.array(x, axis=1)
     print(y)
     [1,2]
    """
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
