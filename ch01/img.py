import matplotlib.pyplot as plt 
from matplotlib.image import imread

img = imread('../ch01/lena.jpg')
plt.imshow(img)
plt.show()