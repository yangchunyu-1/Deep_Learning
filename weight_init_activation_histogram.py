from Sigmoid import sigmoid
import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(1000,100) #1000个数据
node_num = 100
hidden_layer_num = 5 #隐藏层有5层
activations = {}

for i in range(hidden_layer_num):
    if i != 0:
        x = activations[i-1]

    #w = np.random.randn(node_num,node_num) * 1
    w = np.random.randn(node_num,node_num) / np.sqrt(node_num)
    z = np.dot(x,w)
    a = sigmoid(z)
    activations[i] = a

for i,a in activations.items():
    plt.subplot(1,len(activations),i + 1)
    plt.title(str(i + 1) + "-layer")
    plt.hist(a.flatten(),30,range=(0,1))
plt.show()
