# -*- coding: utf-8 -*-
import sys,os

sys.path.append(os.pardir) #mnist.py를 찾으려면 부모 디렉터리로부터 시작해야해서 
from dataset.mnist import load_mnist #dataset/mnist.py 의 load_mnist함수 임포트

"""
처음엔 오래 걸리지만 
pickle(프로그램 실행중에 특정 객체를 파일로 저장하는 기능, 저장해둔 pickle파일을 로드하면 실행 당시의 객체를 즉시 복원)의 기능으로 
2번째 이후의 읽기 시 빠르게 준비 가능
"""
(x_train,t_train),(x_test,t_test) = \
    load_mnist(flatten=True, normalize=False)

#각 데이터의 형상 출력(훈련이미지 60000장, 시험 이미지 10000장)
print(x_train.shape) #(60000,784)
print(t_train.shape) #(60000,)
print(x_test.shape) #(10000,784)
print(t_test.shape) #(10000,)