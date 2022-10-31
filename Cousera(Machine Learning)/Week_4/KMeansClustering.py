import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

snsdata = pd.read_csv("snsdata.csv")
snsdata_clean = snsdata.dropna()
snsdata_clean.describe()

snsdata_clean['gender'] = preprocessing.LabelEncoder().fit_transform(snsdata_clean['gender'])
del snsdata_clean['gradyear'] # drop useless variable

for name in snsdata_clean.columns:
    snsdata_clean[name] = preprocessing.scale(snsdata_clean[name]).astype('float64')
 
from scipy.spatial.distance import cdist

clusters = range(1,10)
meandist = []

for k in clusters:
    model = KMeans(n_clusters = k,random_state = 123)
    model.fit(snsdata_clean)
    clusassign = model.predict(snsdata_clean)
    meandist.append(sum(np.min(cdist(snsdata_clean,model.cluster_centers_,'euclidean'), axis = 1))/snsdata_clean.shape[0])
  
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')
plt.show()

from sklearn.decomposition import PCA

def kmeans(k):
    model = KMeans(n_clusters = k,random_state = 123)
    model.fit(snsdata_clean)
    pca_2 = PCA(2)
    plot_columns = pca_2.fit_transform(snsdata_clean)
    cols = ['r','g','b','y','m','c']
    legentry = []
    legkey = []
    for i in range(k):
        rowindex = model.labels_ == i
        plot_ = plt.scatter(plot_columns[rowindex,0],plot_columns[rowindex,1], c = cols[i],)
        exec('sc' + str(i) + " = plot_")
        legentry.append(eval('sc' + str(i)))
        legkey.append('Cluster ' + str(i + 1))
    plt.legend(tuple(legentry),tuple(legkey),loc = 'lower right')
    plt.xlabel('Canonical variable 1')
    plt.ylabel('Canonical variable 2')
    plt.title('Scatterplot of Canonical Variables for ' + str(k) + ' Clusters')
    plt.show() 

kmeans(3)
kmeans(4)

model3 = KMeans(n_clusters = 3).fit(snsdata_clean)
snsdata_clean.reset_index(level = 0, inplace = True)
newclus = pd.DataFrame.from_dict(dict(zip(list(snsdata_clean['index']),list(model3.labels_))),orient = 'index')
newclus.columns = ['cluster']

newclus.reset_index(level = 0, inplace = True)
snsdata_merge = pd.merge(snsdata_clean,newclus, on = 'index')
snsdata_merge.cluster.value_counts()

clustergrp = snsdata_merge.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)
