# coding: utf-8
import numpy as np
"""
신경망 학습의 목적
-> 손실함수의 값을 가능한 낮추는 매개변수를 찾는 것
-> 이를 최적화라고 함

최적의 매개변수의 값을 찾는 단서로 매개변수의 기울기(미분) 사용
-> 확률적 경사하강법(SGD)이라함.

(SGD의 단점)
기울어진 방향이 본래의 최솟값과 다른방향
즉, 비효율적인 움직임임.

이러한 단점을 개선해주는 
Momentum, AdaGrad, Adam등이 있음
"""
#Momentum
"""
운동량. SGD보다 x축 방향으로 빠르게 다가가 지그재그 움직임이 줄어듬
"""
class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

#v: 물체의 속도
#초기화: 아무것도 담지 않음.
#update(): 매개변수와 같은 구조의 데이터를 딕셔너리 변수로 저장
    def update (self,params,grads):
        if self.v is None:
            self.v={}
            for key,val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys();
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

#AdaGrad
"""
AdaGrad는 개별 매개변수에 적응적으로 학습률을 조정하면서 학습을 진행
처음에는 크게 학습하다가 조금씩 작게 학습한다는 얘기
-> 크게 갱신된 원수는 학습률이 낮아짐 
"""

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self. h = None

    def update(self,params,grads):
        if self.h is None:
            self.h ={}

            for key,val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -+ self.lr *grads[key]/ (np.sqrt(self.h[key]) +1e-7) #zero_division_error 방지


#Adam
"""
momentum + AdaGrad
"""

