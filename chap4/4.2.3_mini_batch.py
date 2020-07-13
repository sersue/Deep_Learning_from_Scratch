"""
훈련데이터 60000개의 모든 데이터의 손실함수의 합을 구하면
시간이 너무 오래걸림
그래서 보통 데이터 일부를 추려 전체의 근사치로 이용 --> 미니배치

신경망 학습에서 정확도를 지표로 삼는 것이 아니라 손실함수를 지표로 삼아야함
(이유)
정확도를 지표로 삼으면 계단함수와 같이 대부분의 장소에서 매개변수의 미분 값이 0이 됨
하지만 손실함수는 시그모이드 함수로 값이 연속적으로 변화함.
"""


import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train),(x_test ,t_test) =\
    load_mnist(normalize=True,one_hot_label =True)

print(x_train.shape) #(60000,784)
print(t_train.shape) #(60000, 10)

#지정한 수의 데이터를 무작위로 골라내는 코드

train_size = x_train.shape[0]
batch_size =10
batch_mask = np.random.choice(train_size,batch_size) #0~ 60000미만 중에 무작위로 10개 골라냄 (이 함수가 출력한 배열을 미니배치로 뽑아낼 데이터의 인덱스로 사용)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# 데이터 하나(배열의 차원수 .ndim)당 교차 엔트로피 오차를 구하는 경우
def cross_entropt_error(y,t):
    if y.ndim ==1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7)) /batch_size

# 정답 레이블이 원핫-인코딩[0,0,1,0,0]이 아니라 2나 7등의 숫자 레이블로 주어졌을 때의 교차엔트로피 오차를 구하는 경우
def cross_entropt_error(y,t):
    if y.ndim ==1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) /batch_size
