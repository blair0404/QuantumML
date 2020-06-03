#!/usr/bin/env python
# coding: utf-8

# In[633]:


from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urlretrieve(iris)
df = pd.read_csv(iris, sep=',')
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df.columns = attributes


# In[636]:


df.head()


# In[638]:





# In[639]:


import math 
from math import pi

#import Qiskit
from qiskit import Aer, IBMQ, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

#import basic plot tools
from qiskit.tools.visualization import plot_histogram


# In[640]:


# To use local qasm simulator
backend = Aer.get_backend('qasm_simulator')
backend 


# In[641]:


def get_theta(d):
    x = d[0]
    y = d[1]
    
    theta = 50*math.acos((x+y)/50) 
    return theta
#get_theta([0.3,0.4])


# In[642]:


def get_Distance(x,y):
    theta_1 = get_theta(x)
    theta_2 = get_theta(y)
    
    # create Quantum Register called "qr" with 3 qubits
    qr = QuantumRegister(3, name="qr")
    # create Classical Register called "cr" with 5 bits
    cr = ClassicalRegister(3, name="cr")

    # Creating Quantum Circuit called "qc" involving your Quantum Register "qr"
    # and your Classical Register "cr"
    qc = QuantumCircuit(qr, cr, name="k_means")
    
    qc.h(qr[0])
    qc.h(qr[1])
    qc.h(qr[2])
    qc.u3(theta_1, pi, pi, qr[1])
    qc.u3(theta_2, pi, pi, qr[2])
    qc.cswap(qr[0], qr[1], qr[2])
    qc.h(qr[0])

    qc.measure(qr[0], cr[0])
    #print(qc.measure(qr[0], cr[0]))
    qc.reset(qr)

    #print('----before run----')
    job = execute(qc,backend=backend, shots=1024)
    #print('----after run----')
    result = job.result()
    data = result.data()['counts']
    #print(data)
    if len(data)==1:
        return 0.0
    else:
        return data['0x1']/1024.0


# In[643]:


def get_center(n,k):
    data = make_blobs(n_samples=n, n_features=2, centers=k, cluster_std=std, random_state=100)
    points = data[0]
    centers = data[1]
    
    return centers
#len(get_center(140,5))


# In[644]:


def draw_plot(points,centers,label=True):
    if label==False:
        plt.scatter(points[:,0], points[:,1])
    else:
        plt.scatter(points[:,0], points[:,1], c=centers, cmap='viridis')
    #plt.xlim(0,10)
    #plt.ylim(0,10)
    plt.show()


# In[645]:


def plot_centroids(centers):
    plt.scatter(centers[:,0], centers[:,1])
    #plt.xlim(0,10)
   # plt.ylim(0,10)
    plt.show()


# def initialize_centers(points,k):
#     return points[np.random.randint(points.shape[0],size=k),:]

# def get_distance(p1, p2):
#     return np.sqrt(np.sum((p1-p2)*(p1-p2)))

# In[646]:


def find_nearest_neighbour(points,centroids):
    
    n = len(points)
    k = centroids.shape[0]
    centers = np.zeros(n)
    
    for i in range(n):
        min_dis = 10000
        ind = 0
        for j in range(k):
            
            temp_dis = get_Distance(points[i,:],centroids[j,:])
            #print(temp_dis)
            
            if temp_dis < min_dis:
                min_dis = temp_dis
                ind = j
        centers[i] = ind
    
    return centers


# In[647]:


def find_centroids(points,centers):
    n = len(points)
    k = int(np.max(centers))+1
   
    centroids = np.zeros([k,2])
    
    for i in range(k):
        #print(points[centers==i])
        centroids[i,:] = np.average(points[centers==i])
        
    return centroids


# def preprocess(points):
#     n = len(points)
#     x = 30.0*np.sqrt(2)
#     for i in range(n):
#         points[i,:]+=15
#         points[i,:]/=x
#     
#     return points

# In[648]:


# x和y的值
data_x = df["sepal_length"]
data_y = df["sepal_width"]

#我们需要的两个feature，data
data = pd.concat([data_x,data_y], axis=1, sort=False)

#随机生成最开始的k个centroid
a = data.sample(n=k, random_state= 888)

#转变成np array
initial = np.asarray(a)
dataset = np.asarray(data)

#n = size
size= df.shape[0]

#输入需要的值
n = size     # number of data points
k = 5        # Number of centers
#std = 2      # std of datapoints



o_center = get_center(n,k)       #dataset

#points = preprocess(points)                # Normalize dataset
#plt.figure()                                  
draw_plot(dataset,o_center)

centroids = initial   # Intialize centroids

converge = 0
# run k-means algorithm
for i in range(50):
    
    centers = find_nearest_neighbour(dataset,centroids)       # find nearest centers
    plt.figure()
    draw_plot(dataset,centers)
    #plot_centroids(centroids)
    past_centroids = centroids
    centroids = find_centroids(dataset,centers)               # find centroids
    diff = centroids - past_centroids
    if diff.any() < 0.1:
        break
    converge += 1
print("Converge at: "+ str(converge))


# In[ ]:




