import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans 
mall_data = pd.read_csv("C:/Users/SL3-18/Downloads/Mall_Customers.csv") 
mall_data.head() 
#values 
mall_data.info() 
# We can use a heatmap to check correlation between the variables. 
corr = mall_data.corr() 
plt.figure(figsize=(8,8)) 
sns.heatmap(corr,cbar=True,square=True,fmt='.1f',annot=True,cmap='Reds') 
# People of what ages shop more? 
plt.figure(figsize=(16,10)) 
sns.countplot(x="Age", data=mall_data) 
# Which gender shops more? 
plt.figure(figsize=(10,10)) 
sns.countplot(x="Gender", data=mall_data) 
# Is there really no relationship between annual income and spending score? 
plt.figure(figsize=(20,8)) 
sns.barplot(x='Annual Income (k$)',y='Spending Score (1-100)',data=mall_data) 
# For our model, we can choose those variables which are relevant or necessary, we needn't choose 
all. 
# Choose age, annual income and spending score columns for clustering model. 
X = mall_data.iloc[:,[2,3,4]].values 
#Now, we need to find the optimal number of clusters for this dataset.  
#To do that, we will use WCSS (within-cluster-sum-of-squares) parameter.  
#WCSS is the sum of squares of the distances of each data point in all clusters to their respective 
centroids.  
#The idea is to minimise the sum. 
wcss = [] 
for i in range(1,11): # It will find wcss value for different number of clusters (for 1 cluster, for 2...until 
10 clusters) and put it in our list 
kmeans = KMeans(n_clusters=i, init='k-means++', random_state=50) 
kmeans.fit(X) 
wcss.append(kmeans.inertia_) 
wcss 
# elbow graph 
sns.set() 
plt.plot(range(1,11),wcss) 
plt.xlabel("Number of clusters") 
plt.ylabel("WCSS value") 
plt.show() 
#Building the model 
kmeans = KMeans(n_clusters = 5, init = 'k-means++',random_state = 0) 
# we need a label for each datapoint relative to their clusters  
#(will be split into 5 clusters and each will be labelled 0-4) 
y = kmeans.fit_predict(X) 
# 3d scatterplot using matplotlib 
f
 ig = plt.figure(figsize = (10,10)) 
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(X[y == 0,0],X[y == 0,1],X[y == 0,2], s = 40 , color = 'red', label = "cluster 1") 
ax.scatter(X[y == 1,0],X[y == 1,1],X[y == 1,2], s = 40 , color = 'blue', label = "cluster 2") 
ax.scatter(X[y == 2,0],X[y == 2,1],X[y == 2,2], s = 40 , color = 'green', label = "cluster 3") 
ax.scatter(X[y == 3,0],X[y == 3,1],X[y == 3,2], s = 40 , color = 'yellow', label = "cluster 4") 
ax.scatter(X[y == 4,0],X[y == 4,1],X[y == 4,2], s = 40 , color = 'purple', label = "cluster 5") 
ax.set_xlabel('Age of a customer-->') 
ax.set_ylabel('Anual Income-->') 
ax.set_zlabel('Spending Score-->') 
ax.legend() 
plt.show() 
OUTPUT :- 
Which gender shops more- 
People of what ages shop more- 
Is there really no relationship between annual income and spending score- 
the optimal number of clusters for this dataset- 
[308812.77999999997, 
212840.16982097182, 
143342.751571706, 
104366.15145556198, 
75350.77917248776, 
58300.44332159069, 
51118.9499316473, 
44637.395640170645, 
40936.763713763714, 
37048.591333481345] 
elbow graph- 
3d scatterplot using matplotlib- 
