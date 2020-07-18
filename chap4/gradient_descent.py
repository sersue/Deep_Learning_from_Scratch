# -*- coding: utf-8 -*-

"""
경사하강법 (경사법)

기계학습은 학습단계에서 최적의 매개변수를 찾아냄
-> 최적의 매개변수 ?
손실함수가 최솟값이 되는 매개변수의 값.

일반적인 문제의 손실합수는 복잡
-------------------------------

이런상황에서 기울기를 이용해 함수의 최솟값을 찾으려는 것이 경사법
"""

def numerical_gradient(f,x):
    h = 1e-2
    grad = np.zeros_like(x)

    
    for idx in range(x.size):
        tmp_val = x[idx]
    #f(x+h)
        x[idx] = tmp_val +h
        fxh1 =f(x)
    #f(x-h)
        x[idx] = tmp_val -h
        fxh2 = f(x)
    
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    
    return grad
"""
lr : 학습률 (한번의 학습으로 얼마만큼 학습해야할지, 매개변수 값을 얼마나 갱신하는지)
   : 0.01이나 0.001 등 미리 특정값으로 정해두어야함
   : 값이 너무 작거나 크면 안됨
"""
def gradient_descent(f,init_x,lr = 0.01, step_num=100):
    x = init_x
    for i in range (step_num):
        grad = numerical_gradient(f,x)
        x -= lr *grad
        
    return x 