#This script aims to classify transactions as fraudulent or non-fraudulent, based on a number of variables
# Uses XGBoost Classifier.
#Transaction data was sourced from Kaggle

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
import pickle
from imblearn.combine import SMOTEENN

def test_run():
	#read data
	data = pd.read_csv('/Desktop/creditcard.csv')

	#get some correlations
	corr_matrix = data.corr()
	print('Correlations')
	print(corr_matrix["Class"].sort_values(ascending=False))

	#select predictor variables and drop missing data
	df = data.loc[:, data.columns != 'Class']
	df.dropna()

	#assign target variable (Class in this case)
	target = pd.DataFrame(data, columns=["Class"])

	X = df
	y = target

	#solve dataset imbalances on dependent variable using SMOTEENN algorithm
	#sme = SMOTEENN(random_state=42)
	#X, y = sme.fit_sample(X, y)

	# Standardize features
	scaler = StandardScaler()
	X_std = scaler.fit_transform(X)

	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_std, y, test_size=0.3, random_state=42)

	#Create model
	clf = XGBClassifier(max_depth=6, min_child_weight = 1, eta=0.1, silent=1, objective='multi:softmax', num_class=2) 

	# Train model
	model = clf.fit(X_train, Y_train.values.ravel())

	# make predictions for test data
	y_pred = model.predict(X_test)
	predictions = [round(value) for value in y_pred]
	#print(predictions)

	print('Classification Report')
	print (classification_report(Y_test, predictions))

	#confusion matrix
	print('Confusion Matrix')
	print(confusion_matrix(Y_test, predictions))

	#k fold validation
	kfold = StratifiedKFold(n_splits=10, random_state=7) 
	results = cross_val_score(clf, X_std, y, cv=kfold)
	print("Stratified K-Fold Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

	# plot feature importance
	plot_importance(model)
	plt.show()

	# save model
	filename = '/Desktop/Credit_model.pkl'
	pickle.dump(model, open(filename, 'wb'))

	# predict values in original data, to see how our model's predictions compare with real values
	b = model.predict(X_std)

	# send predictions to csv, after merging them with original data
	df2 = pd.DataFrame(data={"predicted":b})
	pd.set_option('display.max_colwidth', -1)

	data['Predicted'] = df2

	data.to_csv(r"/Desktop/predicted.csv")

if __name__ == '__main__':
	test_run()	
