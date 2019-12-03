import numpy as np

import matplotlib
from matplotlib import pyplot as plt

import matplotlib.gridspec as gs
x_axis =[1,2,3,4,5,6,7,8,9,10]
y1_0 = [ 0.2411,0.3460,0.4168,0.4712,0.5188,0.5559,0.5887,0.6190,0.6450,0.6645]
y2_0 = [0.2419 ,0.3547,0.4168 ,0.4767,0.5204,0.5601,0.5890,0.6208 ,0.6490 ,0.6734]
y3_0 = [0.2800,0.3978, 0.4830 ,0.5309,0.5722,0.6072,0.6348,0.6658 ,0.6860 ,0.7050]
y4_0 = [0.3061,0.4283,0.5007, 0.5625,0.6161,0.6600,0.6939,0.7186,0.7426,0.7607]
y5_0 = [0.2351,0.3487,0.4307,0.4933 ,0.5488,0.5880,0.6177,0.6458,0.6729,0.6979]
fig2 = plt.figure(figsize=(45,15))
fig2.tight_layout()
subplot = plt.subplot(1,3,1)

subplot.plot(x_axis,y1_0, color= "red", linewidth= 2 ,marker ='*',mec='r',ms=10,)
subplot.plot(x_axis,y2_0, color= "blue", linewidth= 2,marker ='o',mec='blue',ms=10,)
subplot.plot(x_axis,y3_0, color= "black", linewidth= 2,marker ='>',mec='black',ms=10,)
subplot.plot(x_axis,y4_0, color= "darkorange", linewidth= 2,marker ='D',mec='darkorange',ms=10,)
subplot.plot(x_axis,y5_0, color= "green", linewidth= 2,marker ='p',mec='green',ms=10,)
plt.xlabel('TopK')
my_x_ticks = np.arange(0,11,1)
my_y_ticks = np.arange(0,1,0.05)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)

plt.show()