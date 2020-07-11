import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) # 넘파이 배열 --> 이미지 객체로 변환 
    pil_img.show()

(x_train,t_train),(x_test,t_test) = \
    load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0] #그림에 붙여있는 정답
print(label)

print(img.shape) #(784,)
img = img.reshape(28,28) #flatten = True로 설정해 읽어 들인 이미지는 1차원 넘파이 배열로 저장 되어있음 --> reshpe()메서드로 원하는 형상을 인수로 저장
print(img.shape) #(28,28)

img_show(img)