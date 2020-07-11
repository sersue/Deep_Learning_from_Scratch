# -*- coding: utf-8 -*-
# 최근에는 활성화 함수로 ReLU 함수를 주로 이용
# 입력이 0을 넘으면 그 입력을 그대로 출력, 0 이하이면 0을 출력하는 함수
import numpy as np

def relu(x):
    return np.maximum(0,x)


