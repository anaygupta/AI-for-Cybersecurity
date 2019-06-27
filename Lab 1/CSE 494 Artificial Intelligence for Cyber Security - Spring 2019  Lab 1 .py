#!/usr/bin/env python
# coding: utf-8

# # K Means Clustering
# 
# The $K$-means algorithm divides a set of $N$ samples $X$ into $K$ disjoint clusters $C$, each described by the mean $\mu_j$ of the samples in the cluster. The means are commonly called the cluster “centroids”; note that they are not, in general, points from $X$, although they live in the same space. The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum of squared criterion:
# 
# $$\sum_{i=0}^{n}\min_{\mu_j \in C}(||x_j - \mu_i||^2)$$
# 
# ### How the algorithm works
# 
# The Κ-means clustering algorithm uses iterative refinement to produce a final result. The algorithm inputs are the number of clusters $Κ$ and the data set. The data set is a collection of features for each data point. The algorithms starts with initial estimates for the $Κ$ centroids, which can either be randomly generated or randomly selected from the data set. The algorithm then iterates between two steps:
# 
# $\textbf{Data assigment step}$: Each centroid defines one of the clusters. In this step, each data point is assigned to its nearest centroid, based on the squared Euclidean distance. More formally, if $c_i$ is the collection of centroids in set $C$, then each data point $x$ is assigned to a cluster based on
# 
# $$\underset{c_i \in C}{\arg\min} \; dist(c_i,x)^2$$
# where dist( · ) is the standard ($L_2$) Euclidean distance. Let the set of data point assignments for each ith cluster centroid be $S_i$.
# 
# $\textbf{Centroid update step}$: In this step, the centroids are recomputed. This is done by taking the mean of all data points assigned to that centroid's cluster.
# 
# $$c_i=\frac{1}{|S_i|}\sum_{x_i \in S_i x_i}$$
# The algorithm iterates between steps one and two until a stopping criteria is met (i.e., no data points change clusters, the sum of the distances is minimized, or some maximum number of iterations is reached).
# 
# ### Convergence and random initialization
# 
# This algorithm is guaranteed to converge to a result. The result may be a local optimum (i.e. not necessarily the best possible outcome), meaning that assessing more than one run of the algorithm with randomized starting centroids may give a better outcome.
# 
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


df = pd.read_csv('../../data/lab1/College.csv')
df


# In[6]:


df_orig = pd.read_csv('../../data/lab1/College.csv')


# In[7]:


df.head()


# In[8]:


df.size


# In[9]:


df.shape


# In[10]:


df.info()


# ## The Data
# 
# We will use a data frame with $777$ observations on the following 18 variables.
# 
# 
# -  Private -  A factor with levels No and Yes indicating private or public university
# -  Apps - Number of applications received
# -  Accept -  Number of applications accepted
# -  Enroll - Number of new students enrolled
# -  Top10perc -  Pct. new students from top 10% of H.S. class
# -  Top25perc -  Pct. new students from top 25% of H.S. class
# -  F.Undergrad -  Number of fulltime undergraduates
# -  P.Undergrad - Number of parttime undergraduates
# -  Outstate -  Out-of-state tuition
# -  Room.Board - Room and board costs
# -  Books - Estimated book costs
# -  Personal -  Estimated personal spending
# -  PhD - Pct. of faculty with Ph.D.’s
# -  Terminal - Pct. of faculty with terminal degree
# -  S.F.Ratio - Student/faculty ratio
# -  perc.alumni -  Pct. alumni who donate
# -  Expend - Instructional expenditure per student
# -  Grad.Rate -  Graduation rate
# 

# In[11]:


df.columns


# ## Selection and Indexing Methods for Pandas DataFrames
# 
# -  The iloc indexer for Pandas Dataframe is used for integer-location based indexing / selection by position.
# 
# > The iloc indexer syntax is $ \textbf{data.iloc[<row selection>, <column selection>]}$

# In[12]:


df.iloc[0] # this is a Series output


# In[13]:


df.iloc[:, 1] # # second column of data frame (Private)


# In[14]:


df.iloc[[0,3,6,24], [0,5,6]] # 1st, 4th, 7th, 25th row + 1st 6th 7th columns.


# ## Selecting pandas data using “loc”
# 
# The Pandas loc indexer can be used with DataFrames for two different use cases:
# 
# -  Selecting rows by label/index
# -  Selecting rows with a boolean / conditional lookup
df.loc[1, :]
# ### Rename Unnamed column - the university names

# In[15]:


df.rename( columns={'Unnamed: 0':'University'}, inplace=True )


# In[16]:


df.head()


# In[43]:


df.set_index('University', inplace=True)

df.loc['Arizona State University Main campus']df.reset_index
df.iloc[:10]df.head()
# ### Setting values in DataFrames using .loc
# 
# With a slight change of syntax, you can actually update your DataFrame in the same statement as you select and filter using .loc indexer. This particular pattern allows you to update values in columns depending on different conditions. The setting operation does not make a copy of the data frame, but edits the original data.

# In[17]:


df.describe()


# ## Preparing the labels for the instances
# 
# -  For this lab, for the purposes of cluster evaluation, we will use one of the columns as a held-out variable.
# -  We will use the Grad.Rate column and bucket them into various categories to check whether the clusters can be predictive of the graduation rates of the schools

# In[18]:


df.hist(column='Grad.Rate')


# In[108]:


def bin_gradRate(x):
    if x<40:
        return 'Low' # low
    if x>=40 and x<60:
        return 'Medium' # medium
    if x>=60 and x<80:
        return 'High' # high
    else:
        return 'Very High' # very high

df['Grad.Rate.Category'] = df['Grad.Rate'].apply(lambda x: bin_gradRate(x))df.head()
# In[20]:


df['Grad.Rate.Bins'] = pd.cut(df['Grad.Rate'], 6, labels=list(range(6))) # Bins/Discretize the values into intervals


# In[21]:


# df = df.drop('Grad.Rate', axis=1)


# In[22]:


df.head()

df.dtypes

# In[23]:


df['Grad.Rate.Bins'].value_counts()


# ## Encoding categorical features
# 
# -  In this lab, we will focus on one-hot encoding the categorical features using the Pandas in-built functions
# -  Alternatively, scikit-learn has methods for encoding categorical features https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features or Scikit-Learns's DictVectorizer https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html

# In[259]:


# use pd.concat to join the new columns with your original dataframe
df_encode = pd.concat([df,pd.get_dummies(df['Private'], prefix='private')],axis=1)

df_encode.head()
# In[24]:


df.Private[df.Private == 'Yes'] = 1
df.Private[df.Private == 'No'] = 0


# In[25]:


df.head()


# #### Store the schools in a dictionary and drop that column

# In[26]:


schools_dict = df['University'].to_dict()
schools_dict


# In[283]:


df = df.drop('University', axis=1)

df.head()corr = pd.DataFrame()

list_target = ['Grad.Rate.Bins']
list_to_correlate = list(df.columns.values)[:-1]



df_corr = df.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)

for a in list_target:
    for b in list_to_correlate:
        corr.loc[a, b] = df_corr.loc[a, b]

corr

# In[344]:


columns_to_select = ['Top10perc', 'Top25perc', 'perc.alumni', 'Enroll', 'Grad.Rate.Bins' ]

df_sel = df[columns_to_select]
df_sel.head()
# ### Scikit learn user guide on clustering algorithms
# 
# To read more on K-Means implementation or other clustering algorithms, you can see the user guide at https://scikit-learn.org/stable/modules/clustering.html#k-means
# 
# $ \textit{class sklearn.cluster.KMeans}$(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto')  
# 
# Some of the parameters we would use in this lab or for homeworks:
# 
# - n_clusters
# - init
# - max_iter
# 
# Returns:
# 
# - Labels
# - Inertia
# 
# 
# 
# 

# #### Create an instance of a K Means model with 6 clusters
# 

# In[297]:


from sklearn.cluster import KMeans

# The K Means fit function in scikit learn
kmeans = KMeans(n_clusters=6,init='random', verbose=0,tol=1e-3,max_iter=300,n_init=10)


# #### Fit the model to all the data except for the Gard.Rate.Category
df = df.drop('Grad.Rate', axis=1) # Remove this later
df.iloc[:10]df.head()kmeans.fit(df_sel)

# #### What are the cluster center vectors?
# 
# 
clus_cent=kmeans.cluster_centers_
clus_cent
# #### Create a data frame with cluster centers and with column names borrowed from the original data frame
#df_desc=pd.DataFrame(df.describe())
# df_temp = df.drop('Grad.Rate.Category', axis=1)
feat = list(df_sel.columns)
kmclus = pd.DataFrame(clus_cent,columns=feat)
kmclus
df[df['Grad.Rate.Category']==0].describe()
# #### We want to measure the distance between a cluster center and a sample using Euclidean distance:
# 
# The distance is defined as:
# 
# \begin{equation}
# d(p,q)= \sqrt{\sum_{i=1}^n (p_i - q_i)^2 }
# \end{equation}
# 
# 

# In[303]:


def _distance(a, b):
    return np.sqrt(((a - b)**2).sum())


# In[304]:


def _nearest(clusters, x):
    return np.argmin([_distance(x, c) for c in clusters])

df.head()
# In[305]:


# Index for Arizona State University
for idx, s in schools_dict.items():    
#     print(s)
    if s == 'Arizona State University Main campus':
        print(idx)


# #### Get the feature values for Arizona State University Main campus
# 

# In[319]:


feat_asu = df_sel.iloc[23]


# In[320]:


feat_asu_arr = feat_asu.values
# feat_asu_arr = feat_asu_arr[:-1]


# In[321]:


# df_orig = pd.read_csv('../../data/lab1/College.csv')
# df_orig = df_orig.rename(columns = {'Unnamed: 0': 'Schools'})

df_orig[df_orig['Schools'] == 'Arizona State University Main campus']
# In[322]:


clus_cent.shape


# In[323]:


feat_asu_arr


# In[324]:



_nearest(clus_cent, feat_asu_arr.T)

kmeans.labels_
# ## Silhouette Analysis
# 
# Silhouette analysis can be used to study the separation distance between the resulting clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters and thus provides a way to assess parameters like number of clusters visually. This measure has a range of $[-1, 1]$.
# 
# $\textbf{Silhouette coefficients}$ (as these values are referred to as) near +1 indicate that the sample is far away from the neighboring clusters. A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters and negative values indicate that those samples might have been assigned to the wrong cluster.
# 
# ### Silhouette Score
# 
# Compute the mean Silhouette Coefficient of all samples.
# 
# The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.
# 
# This function returns the mean Silhouette Coefficient over all samples.
# 
# The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.

# In[355]:


from sklearn.metrics import silhouette_samples, silhouette_score

X = df_sel.values

print(silhouette_score(X, kmeans.labels_))


# ### Scaling the features
hist = df_sel.hist(bins=10)
# For this experiment you are going to take 0 - 1 as the uniform value range across all the features.
# 
# Look at Scikit learn's MinMax scaler https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
# 
from sklearn.preprocessing import MinMaxScaler

X = df_sel.values

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
X_scaledkmeans.fit(X_scaled)
# ### Initializing own cluster centers
# 
# As an example, let us initialize the cluster centers for the 6 clusters using the mode of the data points in each Grad.Rate.Bins interval
df.head()from scipy import stats

custom_cluster_centers = np.zeros((6, len(columns_to_select)))
for i in range(6):
    df_gr = df_sel[df_sel['Grad.Rate.Bins'] == i]
    
    custom_cluster_centers[i, :] = stats.mode(df_gr.values)[0][0]

custom_cluster_centers = custom_cluster_centers[:, :-1]
print(custom_cluster_centers)
# In[356]:


kmeans = KMeans(n_clusters=6,init=custom_cluster_centers, verbose=0,tol=1e-3,max_iter=300,n_init=10)


# In[357]:


kmeans.fit(X_scaled)

print(silhouette_score(X_scaled, kmeans.labels_))

plt.figure(figsize=(10, 7))  
plt.scatter(X_scaled[:,1], X_scaled[:,2], c=kmeans.labels_, cmap='rainbow') # Inertia -  the sum of squared distances of samples to the nearest cluster centre
K = range(2, 15)
inertia_list = []

for i in range(2, 15):
    kmeans = KMeans(n_clusters=i, verbose=0,tol=1e-3,max_iter=300,n_init=10)
    kmeans.fit(X_scaled)
    inertia_list.append(kmeans.inertia_)

plt.plot(K, inertia_list, marker='o')
plt.xlabel('k')
plt.show()

# ### Evaluating when the labels are present
# 

# Formally, we define precision and recall for a set of clusters C and a set of classes Y as:
# 
# $$
# \begin{equation}
#     Precision = \frac{1}{n} \sum_{c \in C} \#_c
# \end{equation}
# $$
# 
# and 
# 
# $$
# \begin{equation}
#     Recall = \frac{1}{n} \sum_{y \in Y} \#_y
# \end{equation}
# $$
# 
# and 
# 
# $$
# \begin{equation}
# F1 = \frac{2PR}{P + R}
# \end{equation}
# $$
# 
# Here $\textbf{n}$ is the number of instances, $\textbf{C}$ denote the  clusters, $\textbf{Y}$ denote the  classes.
# Here, $ \#_c$ is the largest number of instances in cluster c sharing the same class and $ \#_y $ the largest number of instances labeled y within one cluster.

# ## Hierarchical clustering in scikit-learn
# 
# The AgglomerativeClustering object performs a hierarchical clustering using a bottom up approach: each observation starts in its own cluster, and clusters are successively merged together. The linkage criteria determines the metric used for the merge strategy:
# 
# - $\textbf{Ward}$ minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.
# - $\textbf{Maximum}$ or complete linkage minimizes the maximum distance between observations of pairs of clusters.
# - $\textbf{Average linkage}$ minimizes the average of the distances between all observations of pairs of clusters.
# - $\textbf{Single linkage}$ minimizes the distance between the closest observations of pairs of clusters.
# 
# $\textbf{Single linkage}$: measures the closest pair of points
# 
# 
# \begin{equation}
# d_{single}(G, H) = min_{i \in G, j \in H} d_{ij}
# \end{equation}
# 
# 

# In[229]:


import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))  
plt.title("School Dendograms")  
dend = shc.dendrogram(shc.linkage(df.values[:100], method='single'))  
# In[383]:


from sklearn.cluster import AgglomerativeClustering

data = df_sel[:100].values
cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')  
cluster_fit = cluster.fit(data)  

plt.figure(figsize=(10, 7))  
plt.scatter(data[:,1], data[:,2], c=cluster.labels_, cmap='rainbow') 
# In[390]:


# Function to convert the output of scikit-learn's AgglomerativeClustering into the linkage matrix required by
# scipy's dendrogram function
# It takes in the model fit by AgglomerativeClustering, plus all the usual arguments of the dendrogram
# function: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
# Original by Mathew Kallada (BSD 3 licence), https://github.com/scikitlearn/scikit-learn/pull/3464/files
# Original computes numbers of children incorrectly
# Fixed by Derek Bridge 2017
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    tree_as_list = model.children_
    sizes = {}
    linkage_array = []
    start_idx = len(tree_as_list) + 1
    idx = start_idx
    for children in tree_as_list:
        linkage = []
        size = 0
        for child in children:
            linkage += [child]
            if child < start_idx:
                size += 1
            else:
                size += sizes.get(child)
        linkage += [idx - start_idx + 1, size]
        sizes[idx] = size
        idx += 1
        linkage_array += [linkage]
    dendrogram(np.array(linkage_array).astype(float), **kwargs)

fig = plt.figure(figsize=(10,20))
plot_dendrogram(cluster_fit, orientation="left", leaf_font_size=10)
plt.show()
# ### Agglomerative Clustering: Discussion
# 
# 1. You don't have to run the algorithm to completion. You could exit early: <br>
# 
#     1.1 when you have a certain number of clusters, or  <br>
#     1.2 when the next merge would result in a 'bad' cluster, using some measure such as max distance within a cluster.<br>
# 
# 2. This algorithm is only suitable for relatively small datasets <br>
# 
#     2.1 You would probably calculate the distance between every pair of objects in advance <br>
#     2.2 But, in every iteration, it compares every cluster with every other <br>
# 
