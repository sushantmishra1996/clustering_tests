
# coding: utf-8

# In[2]:


get_ipython().magic(u'matplotlib notebook')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[3]:


df = pd.read_csv('FRIENDSS07E01.mkv.embedding.txt', header = None, sep=" ")

print(len(df.columns))

colnames = ['timestamp', 'faceID']
for i in range(0,len(df.columns)-2):
    colnames.append(i)
df.columns = colnames
print(df.head())



Xdf = df.drop(['timestamp','faceID'], axis = 1)
print(Xdf.head())



# In[4]:


X = []
for i in range(0,len(Xdf.index)):
    X.append(np.array(Xdf.iloc[i]))
# print(X)
X = np.array(X)
X_lda = X
print(X)
X = preprocessing.scale(X) #comment this if preprocessing after fitting
print(X)


# In[5]:


kmeans = KMeans(n_clusters=6, n_jobs = -1,max_iter=100 )
kmeans.fit(X)
y = kmeans.labels_
print(y)


# In[6]:


centroids = kmeans.cluster_centers_
print(centroids)
# print(labels)


# In[7]:


#2-dimensional PCA
pca = sklearnPCA(n_components=2)

# if preprocessing done after use this:
# X = preprocessing.scale(X)
# centroids = preprocessing.scale(centroids) 
X_2d = pd.DataFrame(pca.fit_transform(X))
X_2d['labels']=y.tolist()
print(X_2d.head())
cent_2d_pca =pd.DataFrame(pca.fit_transform(centroids))
print(cent_2d_pca.head())


# In[8]:


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
for i in cent_2d_pca.index:
    plt.scatter(cent_2d_pca.iloc[i][0],cent_2d_pca.iloc[i][1],s = 10)
    
plt.show()


# In[11]:


# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',]

for label in set(y):
#     print label
    color = colors[label]
#     print (color)
    for i in range(0,len(X_2d.index)):
        if (X_2d.iloc[i][2] == label):
            plt.scatter(X_2d.iloc[i][0], X_2d.iloc[i][1], marker="o", color=color, s=2, linewidths=5)

plt.legend()
plt.show()

