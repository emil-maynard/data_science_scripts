# In this script, the goal here is to forecast Adidas stockprices, using Adidas price data, as well as the DAX Index price data. 
#Data is stored in a Google Big Query database.
# Model used is ARD Regressor.

import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import ARDRegression
from sklearn import preprocessing, cross_validation, svm
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, Imputer
import matplotlib.pyplot as plt
import seaborn
import math as m
import pickle

def price_forecast():

	#SQL queries to collect data from Google Big Query

	query = "SELECT date, volume, daily_high, daily_low, closing_price, opening_price FROM `dataset.stockprice_collection` WHERE constituent_name = 'ADIDAS AG' ORDER BY date ASC"
	query2 = "SELECT date, volume, daily_high, daily_low, closing_price, opening_price FROM `dataset.stockprice_collection` WHERE constituent_name = 'DAX' ORDER BY date ASC"

	# import data into pandas dataframe
	data = pd.read_gbq(query, project_id="project", index_col=None, col_order=None, reauth=False, verbose=None, private_key="project-key.json", dialect='standard')
	data2 = pd.read_gbq(query2, project_id="project", index_col=None, col_order=None, reauth=False, verbose=None, private_key="project-key.json", dialect='standard')

	data = data[['opening_price', 'daily_high', 'daily_low', 'closing_price', 'volume']]
	data2 = data2[['opening_price', 'daily_high', 'daily_low', 'closing_price', 'volume']]
	
	# rename columns from the second SQL query so we can differentiate between Adidas prices and DAX prices.
	data2.columns = ['DAX_OPEN', 'DAX_HIGH', 'DAX_LOW', 'DAX_CLOSE', "DAX_VOL"]

	print(data.head())
	print(data2.head())

	data = pd.concat([data, data2], axis=1)
	print(data)

	#reset index as numbers spanning the lenght of the dataframe, as current index is set to dates

	data.index = pd.RangeIndex(len(data.index))
	data.index = range(len(data.index))

	# Drop rows with missing values

	data=data.dropna()
	data.Volume = data.volume.astype(float)
	print(data.dtypes)
	print(data.index[-1])

	# Plot the closing price of Adidas

	data.closing_price.plot(figsize=(10,5))
	plt.ylabel("Adidas Close prices")
	plt.show()

	# pick a forecast column
	forecast_col = 'closing_price'

	# Chosing 30 days as number of forecast days
	forecast_out = int(30)
	print('length =',len(data), "and forecast_out =", forecast_out)

	# Creating label by shifting 'Close' according to 'forecast_out'
	data['label'] = data[forecast_col].shift(-forecast_out)
	print(data.head(2))
	print('\n')
	# If we look at the tail, it consists of n(=forecast_out) rows with NAN in Label column
	print(data.tail(2))

	# Define features Matrix X by excluding the label column which we just created
	X = np.array(data.drop(['label'], 1))

	# Preprocessing (impute NAN with column average, and standardize)
	impute = Imputer()
	X = impute.fit_transform(X)
	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	print(X[1,:])

	# X contains last 'n= forecast_out' rows for which we don't have label data
	# Put those rows in different Matrix X_forecast_out by X_forecast_out = X[end-forecast_out:end]

	X_forecast_out = X[-forecast_out:]
	X = X[:-forecast_out]
	print ("Length of X_forecast_out:", len(X_forecast_out), "& Length of X :", len(X))

	# Similarly Define Label vector y for the data we have prediction for
	# A good test is to make sure length of X and y are identical
	y = np.array(data['label'])
	y = y[:-forecast_out]
	print('Length of y: ',len(y))

	#split into train and test data
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3)

	print('length of X_train and x_test: ', len(X_train), len(X_test))

	# Train, using ARD Regression model
	clf = ARDRegression(n_iter=700)
	model = clf.fit(X_train,y_train)

	# calculate accuracy of test dataset
	accuracy = model.score(X_test, y_test)
	print("Accuracy of model: ", accuracy)

	# Forecast 30 days after the last date in the data, using our Model
	forecast_prediction = model.predict(X_forecast_out)
	print('30-day-forecast')
	print(forecast_prediction)

	# export model
	filename = '/Desktop/ARD_Adidas.pkl'
	pickle.dump(model, open(filename, 'wb'))

	# predict values in original data, to see how our model's predictions compare
	b = model.predict(X)
	#print(b)

	# send predictions and forecasts to csv, after merging them with original data
	df2 = pd.DataFrame(data={"predicted":b})
	df3 = pd.DataFrame(data={"forecasted":forecast_prediction})
	pd.set_option('display.max_colwidth', -1)

	data['predicted'] = df2
	data['forecasted'] = df3
	data

	data.to_csv(r"/Desktop/Adidas_forecastARD.csv")

if __name__ == '__main__':
	price_forecast()
