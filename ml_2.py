import pandas as pd

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

print(nyc.head(3)) #print out the first three rows of the dataframe

from sklearn.model_selection import train_test_split 

#using linear regression estimator 
#the more features you have, the better, so this one won't be very accurate because we are only giving it the date

#date is our independent variable
#we just want the date, the target is the temperature
print(nyc.Date.values) #puts all the the dates into a list, but it is not in one column
print(nyc.Date.values.reshape(-1,1)) #reshape the one dimensional array into two dimensional, with one column 

#give it the data, then the target (temperature)
x_train, x_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1,1), nyc.Temperature, random_state=1)

from sklearn.linear_model import LinearRegression

lr = LinearRegression() #creating a linear regression object 

lr.fit(X=x_train, y=y_train) #x_train is our raw data not our target, y_train is our target - trains are used to train the machine
                             #x's are our data, y's are our targets - tests are used to see the accuracy of the model 
                             #fit method adjusts the slope to make sure the data point distanced are minimalized
                             #coef is m and intercept is b y=mx +b 

print(lr.coef_)
print(lr.intercept_)

predicted = lr.predict(x_test) #give it the variable that you have trained
expected = y_test

for p,e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p:.2f}, expected: {e:.2f}")

predict = (lambda x: lr.coef_ * x + lr.intercept_)

print(predict(2020))
print(predict(1890))
print(predict(2021))

import seaborn as sns

axes = sns.scatterplot(
    data=nyc, 
    x="Date", 
    y="Temperature", 
    hue="Temperature",
    palette='winter', 
    legend=False
)

axes.set_ylim(10,70)

import numpy as np 

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])

print(x)

y = predict(x)

print(y)

import matplotlib.pyplot as plt

line = plt.plot(x,y)
plt.show()


