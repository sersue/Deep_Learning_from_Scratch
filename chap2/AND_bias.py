# -*- coding: utf-8 -*-
import numpy as np
def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    # 편향 : 뉴런이 얼마나 쉽게 활성화 하느냐를 조정하는 매개변수 (theta의 이항 값)
    tmp = np.sum(w*x) + b
    if tmp <=0 :
        return 0
    else:
        return 1

print AND(0,0)
print AND(1,0)
print AND(0,1)
print AND(1,1)
# NAND 와 OR 는 AND와 가중치(w,b)만 다르다
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7 
    tmp = np.sum(w*x) + b
    if tmp <=0 :
        return 0
    else:
        return 1
print NAND(0,0)
print NAND(1,0)
print NAND(0,1)
print NAND(1,1)

def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2 
    tmp = np.sum(w*x) + b
    if tmp <=0 :
        return 0
    else:
        return 1
print NAND(0,0)
print NAND(1,0)
print NAND(0,1)
print NAND(1,1)

# 퍼셉트론은 비선형 XOR를 표현하지 못하므로 NAND ,AND, OR를 사용해서 구현 (다층 퍼셉트론)
def XOR (x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1,s2)
    return y
print XOR(0,0)
print XOR(1,0)
print XOR(0,1)
print XOR(1,1)