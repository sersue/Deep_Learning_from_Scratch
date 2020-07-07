import matplotlib.pyplot as plt 
from matplotlib.image import imread

img = imread('lena.jpg')

plt.imshow(img)
plt.show