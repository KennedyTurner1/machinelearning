import pandas as pd

nyc = pd.read_csv("ave_yearly_temp_nyc_1895-2017.csv")

#print(nyc.head(3)) #print out the first three rows of the dataframe

from sklearn.model_selection import train_test_split 

#using linear regression estimator 
#the more features you have, the better, so this one won't be very accurate because we are only giving it the date

#date is our independent variable
#we just want the date, the target is the temperature
#print(nyc.Date.values) #puts all the the dates into a list, but it is not in one column
#print(nyc.Date.values.reshape(-1,1)) #reshape the one dimensional array into two dimensional, with one column 

#give it the data, then the target (temperature)
train_data, test_data, train_target, test_target = train_test_split(nyc.Date.values.reshape(-1,1), nyc.Value, random_state=11) #data, target column, we all get the same random features

#print(x_test) #taken the array and split it 

from sklearn.linear_model import LinearRegression

lr = LinearRegression() #creating a linear regression object 

lr.fit(X=train_data, y=train_target) #x_train is our raw data not our target, y_train is our target - trains are used to train the machine
                             #x's are our data, y's are our targets - tests are used to see the accuracy of the model 
                             #fit method adjusts the slope to make sure the data point distanced are minimalized
                             #coef is m and intercept is b y=mx +b 
                             #where the learning is taking place 

predicted = lr.predict(test_data) #give it the variable that you have trained
expected = test_target

#for p,e in zip(predicted[::5], expected[::5]):
#    print(f"predicted: {p:.2f}, expected: {e:.2f}")

predict = (lambda x: lr.coef_ * x + lr.intercept_)

import seaborn as sns

axes = sns.scatterplot(
    data=nyc, 
    x="Date", 
    y="Value", 
    hue="Value",
    palette='winter', 
    legend=False
)

axes.set_ylim(10,70)

import numpy as np 

x = np.array([min(nyc.Date.values), max(nyc.Date.values)]) 

#print(x)

y = predict(x)

#print(y)

import matplotlib.pyplot as plt

line = plt.plot(x,y) #plotting min and max temp and date
plt.show()
