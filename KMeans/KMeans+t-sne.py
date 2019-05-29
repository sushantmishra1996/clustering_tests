
# coding: utf-8

# In[6]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


# In[15]:


df = pd.read_csv('FRIENDSS07E01.mkv.embedding.txt', header = None, sep=" ")

print(len(df.columns))

colnames = ['timestamp', 'faceID']
for i in range(0,len(df.columns)-2):
    colnames.append(i)
df.columns = colnames
# print(df.head())



Xdf = df.drop(['timestamp','faceID'], axis = 1)
# print(Xdf.head())



# In[16]:


X = []
for i in range(0,len(Xdf.index)):
    X.append(np.array(Xdf.iloc[i]))
# print(X)
X = np.array(X)


# In[17]:


kmeans = KMeans(n_clusters=6, n_jobs = -1,max_iter=500 )
kmeans.fit(X)
y = kmeans.labels_
y[2] = 4
print(y)
centroids = kmeans.cluster_centers_


# In[18]:


model = TSNE(n_components=2, n_iter = 1000 , learning_rate =200 , angle = 0.5 )
X_2d = pd.DataFrame(model.fit_transform(X))
cent_2d =pd.DataFrame(model.fit_transform(centroids))
X_2d['labels']=y.tolist()


# In[19]:


# print(X_2d)
# print(y)
# print(centroids)


# In[20]:


# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
# colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',]
# print(X_2d)
# for label in set(y):
# #     print label
#     color = colors[label]
# #     print (color)
#     for i in range(0,len(X_2d.index)):
#         if (X_2d.iloc[i][2] == label):
#             ax1.scatter(X_2d.iloc[i][0], X_2d.iloc[i][1], X_2d.iloc[i][2], marker="o", color=color, s=2, linewidths=5)
# plt.show()


# In[21]:


for i in range(0,len(X_2d.index)):
    plt.scatter(X_2d.iloc[i][0], X_2d.iloc[i][1], marker="o", s=5, linewidths=5)
plt.show()


# In[22]:



colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',]
print(X_2d)
for label in set(y):
# #     print label
    color = colors[label]
    for i in range(0,len(X_2d.index)):
        if (X_2d.iloc[i][2] == label):
            plt.scatter(X_2d.iloc[i][0], X_2d.iloc[i][1],color = color, marker="o", s=5, linewidths=5)
plt.show()

