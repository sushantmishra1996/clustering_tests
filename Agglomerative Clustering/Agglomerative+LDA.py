
# coding: utf-8

# In[1]:


#%matplotlib notebook use for 3D visulaization
#use for 2D visulaization
get_ipython().magic(u'matplotlib notebook')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")
from scipy.cluster.hierarchy import dendrogram
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pandas.tools.plotting import parallel_coordinates


# In[2]:


df = pd.read_csv('FRIENDSS07E01.mkv.embedding.txt', header = None, sep=" ")

print(len(df.columns))

colnames = ['timestamp', 'faceID']
for i in range(0,len(df.columns)-2):
    colnames.append(i)
df.columns = colnames
# print(df.head())

Xdf = df.drop(['timestamp','faceID'], axis = 1)
print(Xdf.head())


# In[3]:


X = []
for i in range(0,len(Xdf.index)):
    X.append(np.array(Xdf.iloc[i]))
X = np.array(X)
print(X)


# In[10]:


clf = AgglomerativeClustering(n_clusters=6,linkage= 'ward')
clf.fit(X)
y = clf.labels_
print(y)
#X = preprocessing.scale(X) # No effect of preprocessing before or after fitting


# In[11]:


#2-dimensional LDA
lda = LDA(n_components=2)
lda.fit(X,y)
# print(X)
X_2d = pd.DataFrame(lda.transform(X))
# print(X_2d)
X_2d['labels']=y.tolist()
# print(X_2d.head())


# In[12]:


#Plotting the data point
colors = 10* ['C0', 'C1', 'C2', 'C3', 'C4', 'C5',]

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
            plt.scatter(X_2d.iloc[i][0], X_2d.iloc[i][1], marker=".", color=color, s=50, linewidths=2)
plt.show()


# In[13]:


def plot_dendrogram(model, **kwargs):

    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)
# Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



# In[14]:


plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(clf, labels=clf.labels_)
plt.show()

