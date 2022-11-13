#rock vs mine
#submarine detect it is mine of rock
#work flow
#  (sonar data,
#  data pre-processing,
#  train test data(split), 
# logistic regression model,
#  predition )
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # this is userful to train data 
from sklearn.linear_model import LogisticRegression# this is algoritham which used to detect two different view this is classfifcation model
from sklearn.metrics import accuracy_score #this is used to get accurancy score
#data pre-processing
#------------------------------------------
data = pd.read_csv("sonardata.csv")
print(data.columns)
print(data["R"])
print(data.head(10))
print(data.shape)

data = pd.read_csv("sonardata.csv", header=None)
print(data.describe())#describe statistical measures of the data
print(data[60].unique())#this is showing different value in data 
print(data[60].value_counts())#this is showing how many value in rock and mine

#more data more accurancy
print(data.groupby(60).mean())#it will be giving separtion of data according mine and rock 
x = data.drop(columns=60, axis=1)
y= data[60]

print(x.describe())#this is our traing input values
print(y.describe())#this is our prediction 
#---------------------------------------
#Training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)#test size means how many % data is used for testing
print(x.shape, x_test.shape, x_train.shape)#test data is 10% , train data is 90%
print(y.shape, y_test.shape, y_train.shape)
#this is our training data 
print(x_train)
print(y_train)

#train data is used to train model and test data is used to test data (last preditions)
#-------------------------------------------------
#model training 
#logistic model
model = LogisticRegression()
model.fit(x_train, y_train)#training data is 90% we know already 

#--------------------------------------
#model evaluations(accurancy score)
#fisrt we will find accurancy of x_tain 
x_train_prediction = model.predict(x_train)
trainig_data_accurancy = accuracy_score(x_train_prediction, y_train)#it is measuring acciurancy score of our training data 
print(trainig_data_accurancy)

#second we will finding accurancy of our test data 
x_test_prediction = model.predict(x_test)
test_data_accurancy = accuracy_score(x_test_prediction, y_test)#it is measuring acciurancy score of our test data it will take two input one is input and another is out put 
print(test_data_accurancy)
#------------------------------------------
#use test data for predictions
y_test_1 = model.predict(x_test)
print(y_test_1, "this is our model test data result", np.array(y_test), "this is our original data ")
print(y_test, "this is our original data ")

#--------------------------------------
#by using any random value 
#predictions of out put values taking any value from our data 
input_data = (0.02,	0.0371,	0.0428,	0.0207,	0.0954	,0.0986, 0.1539, 0.1601,	0.3109,	0.2111,	0.1609,	0.1582,	0.2238,	0.0645,	0.066,	0.2273,	0.31,	0.2999,	0.5078,	0.4797,	0.5783,	0.5071,	0.4328,	0.555,	0.6711,	0.6415,	0.7104,	0.808,	0.6791,	0.3857,	0.1307,	0.2604,	0.5121,	0.7547,	0.8537,	0.8507,	0.6692,	0.6097,	0.4943,	0.2744,	0.051,	0.2834,	0.2825,	0.4256,	0.2641,	0.1386,	0.1051,	0.1343,	0.0383,	0.0324,	0.0232,	0.0027,	0.0065,	0.0159,	0.0072,	0.0167,	0.018,	0.0084,	0.009,	0.0032)
#convert your data into numpy array
input_data_array_1 = np.asarray(input_data)
input_data_array = input_data_array_1.reshape(1, -1)
y_prediction = model.predict(input_data_array)
print(y_prediction)









