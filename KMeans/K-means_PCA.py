
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pandas.tools.plotting import parallel_coordinates


# In[32]:


df = pd.read_csv('FRIENDSS07E01.mkv.embedding.txt', header = None, sep=" ")

print(len(df.columns))

colnames = ['timestamp', 'faceID']
for i in range(0,len(df.columns)-2):
    colnames.append(i)
df.columns = colnames
print(df.head())



Xdf = df.drop(['timestamp','faceID'], axis = 1)
print(Xdf.head())



# In[33]:


X = []
for i in range(0,len(Xdf.index)):
    X.append(np.array(Xdf.iloc[i]))
# print(X)
X = np.array(X)
print(X)

# print(X)


# In[34]:


# Y = np.array([[1,2],[3,4],[5,6],[7,8],[14,15],[16,19],[18,20],[18,19]])

kmeans = KMeans(n_clusters=6, n_jobs = -1,max_iter=100 )
kmeans.fit(X)
y = kmeans.labels_
print(y)


# In[35]:


centroids = kmeans.cluster_centers_
print(centroids)
# print(labels)


# In[36]:


pca = sklearnPCA(n_components=2) #2-dimensional PCA
X = preprocessing.scale(X)
centroids = preprocessing.scale(centroids)
X_2d = pd.DataFrame(pca.fit_transform(X))
centroids_2d =pd.DataFrame(pca.fit_transform(centroids))
X_2d['labels']=y.tolist()
# print(X_2d.head)
print(centroids_2d.head())
# X_2d = transformed.as_matrix()
# print(X_2d)
# trans_centroid = pd.Dataframe(pca.fit_transform(centroids))


# In[37]:


for i in centroids_2d.index:
    plt.scatter(centroids_2d.iloc[i][0],centroids_2d.iloc[i][1])
plt.show()


# In[38]:


colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',]

for label in set(y):
#     print label
    color = colors[label]
#     print (color)
    for i in range(0,len(X_2d.index)):
        if (X_2d.iloc[i][2] == label):
            plt.scatter(X_2d.iloc[i][0], X_2d.iloc[i][1], marker="o", color=color, s=10, linewidths=5)

plt.legend()
plt.show()

