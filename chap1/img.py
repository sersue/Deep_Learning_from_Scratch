import matplotlib.pyplot as plt 
from matplotlib.image import imread

img = imread('lena.png') #원래 png 파일이여야 됨..
plt.imshow(img)
plt.show()