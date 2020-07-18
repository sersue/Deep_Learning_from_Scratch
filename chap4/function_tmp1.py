
#편미분
import numpy as np 

def function_2(x):
    return x[0]**2 + x[1]**2

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


