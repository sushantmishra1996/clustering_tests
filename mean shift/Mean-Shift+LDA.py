
# coding: utf-8

# In[38]:


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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pandas.tools.plotting import parallel_coordinates
from sklearn import metrics
from sklearn.metrics import pairwise_distances


# In[39]:


df = pd.read_csv('FRIENDSS07E01.mkv.embedding.txt', header = None, sep=" ")

print(len(df.columns))

colnames = ['timestamp', 'faceID']
for i in range(0,len(df.columns)-2):
    colnames.append(i)
df.columns = colnames
# print(df.head())

Xdf = df.drop(['timestamp','faceID'], axis = 1)
# print(Xdf.head())


# In[40]:


X = []
for i in range(0,len(Xdf.index)):
    X.append(np.array(Xdf.iloc[i]))
X = np.array(X)
print(X)
# X = preprocessing.scale(X) #comment this if preprocessing after fitting


# In[41]:


# quantile =[]
# score =np.array([])
# i=D("0.01");j=0
# while i <= 0.5:
#     quantile.append(i) 
#     i= i + D("0.01"); j+=1
# print(quantile)
# j = 0
# score = pd.DataFrame()
# for q in quantile:
#     bandwidth = estimate_bandwidth(X, quantile=q)
#     if bandwidth!=0:
#         clf = MeanShift(bandwidth = bandwidth)
#         clf.fit(X)
#         L = clf.labels_
#         if int(L.shape[0]) >= 2:
#             score[j][0]= metrics.silhouette_score(X, L, metric='euclidean')
#             score[j][1]= q
#             j+=1
# print(score)
    
# labels = clf.labels_
# print(labels)
# centers = clf.cluster_centers_
# # print(centers)
# # print(clf.get_params)
# print(metrics.silhouette_score(X, labels, metric='euclidean'))


# In[51]:


bandwidth = estimate_bandwidth(X, quantile=0.05)
clf = MeanShift(bandwidth = bandwidth)
clf.fit(X)
L = clf.labels_
score = metrics.silhouette_score(X, L, metric='euclidean')
print(score)
labels = clf.labels_
print(labels)
centers = clf.cluster_centers_


# In[52]:


#2-dimensional LDA
lda = LDA(n_components=2)
lda.fit(X,L)
X_2d = pd.DataFrame(lda.transform(X))
X_2d['labels']=L.tolist()
print(X_2d.head())
# y_cent = np.asarray(['0','1','2','3','4','5'],dtype= float64)
cent_2d =pd.DataFrame(lda.transform(centers))
print(cent_2d.head())


# In[53]:


# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
colors = 100*['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7','C8']

for label in set(labels):
#     print label
    color = colors[label]
#     print (color)
    for i in range(0,len(X_2d.index)):
        if (X_2d.iloc[i][2] == label):
            plt.scatter(X_2d.iloc[i][0], X_2d.iloc[i][1], marker="o", color=color, s=5, linewidths=5)
#             ax1.scatter(X_2d.iloc[i][0], X_2d.iloc[i][1],X_2d.iloc[i][2], marker="o", color=color, s=10, linewidths=5)

plt.legend()
plt.show()


# In[54]:


for i in cent_2d.index:
    plt.scatter(cent_2d.iloc[i][0],cent_2d.iloc[i][1],s = 50)
    
plt.show()

