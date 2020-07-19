# coding: utf-8

"""
오차역전파법
순전파 (forward): 왼쪽에서 오른쪽으로 데이터를 순방향으로 전파
역전파 (backward): 역방향으로 전파함으로써 가중치 매개변수의 기울기를 효율적으로 구함

처리과정 
'계층' : ReLU, Softmax-with-loss, Affine, Softmax 계층으로 구현
      : 모든 계층에서 forward, backward 라는 메서드 구현
      : 동작을 계층으로 모듈화 하면 신경망의 계층을 자유롭게 조합하여 원하는 신경망을 쉽게 만들 수 있음
"""
class MulLayer :
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y

        return out
    
    def backward(self,dout):
        dx = dout *self.y
        dy = dout *self.x

        return dx,dy

apple = 100
apple_num = 2
tax = 1.1

#계층들
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

#순전파
apple_price = mul_apple_layer.forward(apple,apple_num)
price = mul_tax_layer.forward(apple_price,tax)

print(price)

#역전파
dprice = 1
dapple_price ,dtax = mul_tax_layer.backward(dprice)
dapple,dapple_num = num_apple_layer.backward(dapple_price)

prunt(dapple,dapple_num, dtax)