
# coding: utf-8

# In[37]:


#%matplotlib notebook use for 3D visulaization
#use for 2D visulaization
get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from pandas.tools.plotting import parallel_coordinates


# In[38]:


df = pd.read_csv('FRIENDSS07E01.mkv.embedding.txt', header = None, sep=" ")

print(len(df.columns))

colnames = ['timestamp', 'faceID']
for i in range(0,len(df.columns)-2):
    colnames.append(i)
df.columns = colnames
# print(df.head())



Xdf = df.drop(['timestamp','faceID'], axis = 1)
# print(Xdf.head())


# In[47]:


X = []
for i in range(0,len(Xdf.index)):
    X.append(np.array(Xdf.iloc[i]))
X = np.array(X)
Z = X
X = preprocessing.scale(X) #comment this if preprocessing after fitting


# In[40]:


kmeans = KMeans(n_clusters=6, n_jobs = -1,max_iter=100 )
kmeans.fit(X)
y = kmeans.labels_
print(y)



# In[41]:


centroids = kmeans.cluster_centers_
print(centroids)


# In[42]:


# if preprocessing done after fitting, use this:
# X_lda = preprocessing.scale(X_lda)
# centroids_lda = preprocessing.scale(centroids_lda) 


# In[43]:


#2-dimensional LDA
lda = LDA(n_components=2)
lda.fit(X,y)
X_2d = pd.DataFrame(lda.transform(X))
X_2d['labels']=y.tolist()
print(X_2d.head())
# y_cent = np.asarray(['0','1','2','3','4','5'],dtype= float64)
cent_2d =pd.DataFrame(lda.transform(centroids))
print(cent_2d.head())


# In[44]:


#Plotting and Visualization


# In[45]:


#Plotting the centroids
#1.For 3D plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in cent_2d.index:
#     ax.scatter(cent_2d.iloc[i][0],cent_2d_lda.iloc[i][1], cent_2d.iloc[i][2],s = 50)
# plt.show()
    
#2. for 2D plotting
for i in cent_2d.index:
    plt.scatter(cent_2d.iloc[i][0],cent_2d.iloc[i][1],s = 50)
    
plt.show()


# In[46]:


#Plotting the data point
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',]

#1. For 3d Plotting
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
# for label in set(y):
#     color = colors[label]
#     for i in range(0,len(X_2d.index)):
#         if (X_2d.iloc[i][2] == label):
#             ax1.scatter(X_2d.iloc[i][0], X_2d.iloc[i][1],X_2d.iloc[i][2], marker="o", color=color, s=10, linewidths=5)
# plt.show()

for label in set(y):
    color = colors[label]
    for i in range(0,len(X_2d.index)):
        if (X_2d.iloc[i][2] == label):
            plt.scatter(X_2d.iloc[i][0], X_2d.iloc[i][1], marker=".", color=color, s=5, linewidths=2)
plt.show()


# In[155]:


kmeans1 = KMeans(n_clusters=6, n_jobs = -1,max_iter=100 )
kmeans1.fit(Z)
z = kmeans1.labels_
print(z)

Zcentroids = kmeans1.cluster_centers_
# print(Zcentroids)


# In[156]:


#if preprocessing done after fitting, use this:
Z = preprocessing.scale(Z)
Zcentroids = preprocessing.scale(Zcentroids) 
# print(Zcentroids)
# print(Z)


# In[157]:


#2-dimensional LDA
lda = LDA(n_components=2)
lda.fit(Z,z) #Maximizes class separability using z
# print(u)
# print(z)
Z_2d = pd.DataFrame(lda.transform(Z))
Z_2d['labels']=z.tolist()
print(Z_2d.head())
Zcent_2d =pd.DataFrame(lda.transform(Zcentroids))
print(Zcent_2d.head())#if preprocessing done after fitting, use this:


# In[158]:


#2. for 2D plotting.
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',]
for i in Zcent_2d.index:
    plt.scatter(Zcent_2d.iloc[i][0],Zcent_2d.iloc[i][1],color = colors[i],s = 50)
    
plt.show()
#difference happens because there's different fitting of centroids and the data points for maximizing class separability
#in preprocessing after fitting, the centroids are treated as a separately new dataset itself and fitted differently than dataset.


# In[159]:


#Plotting the data point
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',]
for label in set(z):
    color = colors[label]
    for i in range(0,len(Z_2d.index)):
        if (Z_2d.iloc[i][2] == label):
            plt.scatter(Z_2d.iloc[i][0], Z_2d.iloc[i][1], marker=".", color=color, s=5, linewidths=2)
plt.show()

