
# coding: utf-8

# In[104]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
from decimal import Decimal as D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import MeanShift, KMeans, estimate_bandwidth
from sklearn import preprocessing, cross_validation
from sklearn.manifold import TSNE
from pandas.tools.plotting import parallel_coordinates
from sklearn import metrics
from sklearn.metrics import pairwise_distances


# In[105]:


df = pd.read_csv('FRIENDSS07E01.mkv.embedding.txt', header = None, sep=" ")

print(len(df.columns))

colnames = ['timestamp', 'faceID']
for i in range(0,len(df.columns)-2):
    colnames.append(i)
df.columns = colnames
# print(df.head())

Xdf = df.drop(['timestamp','faceID'], axis = 1)
# print(Xdf.head())


# In[106]:


X = []
for i in range(0,len(Xdf.index)):
    X.append(np.array(Xdf.iloc[i]))
X = np.array(X)
print(X)
# X = preprocessing.scale(X) #comment this if preprocessing after fitting


# In[107]:


bandwidth = estimate_bandwidth(X, quantile=0.02)
clf = MeanShift(bandwidth = bandwidth)
clf.fit(X)
L = clf.labels_
score = metrics.silhouette_score(X, L, metric='euclidean')
print(score)
labels = clf.labels_
print(labels)
centers = clf.cluster_centers_
c = clf.cluster_centers_


# In[108]:


model = TSNE(n_components=2, n_iter = 1000 , learning_rate =200 , angle = 0.5 )
X = np.append(X,centers,axis = 0) #
c = len(set(labels))
clabels = np.arange(0,c,1)
# print(len(set(labels)))
# for i in range(0,len(set(labels))):
#     np.append(clabels,i)
#     print(i)
print(clabels)
labels = np.append(labels,clabels)#
X_2d = pd.DataFrame(model.fit_transform(X))
cent_2d =pd.DataFrame(model.fit_transform(centers))
X_2d['labels']=labels.tolist()


# In[109]:


# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
colors = 100*['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7','C8']

for label in set(labels):
#     print label
    color = colors[label]
#     print (color)
    for i in range(0,(len(X_2d.index)-c)):
        if (X_2d.iloc[i][2] == label):
            plt.scatter(X_2d.iloc[i][0], X_2d.iloc[i][1], marker="o", color=color, s=8, linewidths=5)
#             ax1.scatter(X_2d.iloc[i][0], X_2d.iloc[i][1],X_2d.iloc[i][2], marker="o", color=color, s=10, linewidths=5)
    for i in range((len(X_2d.index)-c), len(X_2d.index)):
        if (X_2d.iloc[i][2] == label):
            plt.scatter(X_2d.iloc[i][0], X_2d.iloc[i][1], marker="x", color='k', s=8, linewidths=5)
plt.legend()
plt.show()


# In[76]:


for i in cent_2d.index:
    plt.scatter(cent_2d.iloc[i][0],cent_2d.iloc[i][1],s = 50)
    
plt.show()

