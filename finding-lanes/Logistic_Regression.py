import numpy as np
import matplotlib.pyplot as plt

 
def draw(x1,x2):
  plt.plot(x1,x2)
 
def sigmoid(score):
  return 1/(1+np.exp(-score))
  
n_pts=100
np.random.seed(0)
# to generate same pseudo randomn points
bias= np.ones(n_pts)
# for x0 = 1
top_region=np.array([np.random.normal(10,2,n_pts), np.random.normal(12,2,n_pts), bias]).T
# generating top region points , normal distribution of std_x =2 around point_x = 10 and point_y=12. std_y=2
bottom_region= np.array([np.random.normal(5,2, n_pts), np.random.normal(6,2, n_pts), bias]).T
all_points=np.vstack((top_region, bottom_region))
# vertical stacking all the points
w1=-0.2
w2=-0.35
b=3.5
# some randomn initial weights
line_paramters = np.matrix([w1,w2,b]).T
x1=np.array([bottom_region[:,0].min(), top_region[:,0].max()])
x2= -b/w2 + (x1*(-w1/w2))
# w1x1 + w2x2 = b
 
linear_combination= all_points*line_paramters 
# Theta(T) . X
probabilities= sigmoid(linear_combination)
print(probabilities)    
_, ax= plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0], top_region[:,1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color='b')
draw(x1,x2)
plt.show()