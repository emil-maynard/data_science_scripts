#This is a simple clustering excercise, to cluster the stocks of the SP500, based on daily returns performance, over a period of 7 years.
# 3 clusters will be made, using the Agglomerative CLustering Algorithm
# Data was sourced from Kaggle

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

def SP500_Performance():
	data = pd.read_csv('sandp500/all_stocks_5yr.csv')
	#print(data.info)
	data = pd.DataFrame(data, columns=["date", "close", "Name"])
	data = data.set_index(['Name',data.groupby('Name').cumcount()])['close'].unstack().T
	
	#Transpose data so that the clustering algorithm can read the data and give results correctly.
	#We want to turn the data for each stock into rows of observations
	data = data.T

	#calculate returns on the data so we can cluster based on returns performance
	data = data.pct_change()
	data = data.dropna()
	print(data)

	#standardize data
	scaler = StandardScaler()
	X_std = scaler.fit_transform(data)

	# Create clustering object
	clt = AgglomerativeClustering(linkage='complete', 
                              affinity='euclidean', 
                              n_clusters=3)

	# Train model and generate clusters
	cluster = clt.fit_predict(X_std)

	# show cluster results
	print(cluster)

	#visualise clusters
	plt.figure(figsize=(14, 8))
	plt.scatter(X_std[:, 0], X_std[:, 1], c=cluster, s=50, cmap='viridis')


	# concatenate results to original data and export to csv.
	df6 = pd.DataFrame(data={"Cluster":cluster})
	data = data.reset_index()
	result = pd.concat([data, df6], axis=1, join='inner')
	print(result)
	result.to_csv("Desktop/SP500_Performance.csv")

if __name__ == '__main__':
	SP500_Performance()	

	