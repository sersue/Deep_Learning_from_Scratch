import numpy as np 
import matplotlib.pyplot as plt 

#data 
x = np.arange(0,6,0.1)
y1 = np.sin(x)
y2 = np.cos(x)

#draw graph
plt.plot(x,y1,label="sin")
plt.plot(x,y2,linestyle="--",label="cos")
plt.xlabel("x") #x축 이름
plt.ylabel("y") #y축 이름
plt.title('sin & cos') #title

plt.legend()
plt.show()
