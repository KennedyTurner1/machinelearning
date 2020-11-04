from sklearn.datasets import fetch_california_housing #20648 samples

cali = fetch_california_housing() #bunch object 
                                  #basically a pandas pataframe

#print(cali.DESCR)
'''
print(cali.data.shape) #(20640, 8) rows (samples) and 8 columns (features), data

print(cali.target.shape) #(20640,) just the identifier for each row, target 

print(cali.feature_names) #the names of the columns
                          #['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                          #'Population', 'AveOccup', 'Latitude', 'Longitude']
print(cali.target) #median house value 3.66 is $366,000
'''
import pandas as pd 

pd.set_option("precision", 4) #4 digits for floats
pd.set_option("max_columns", 9) #up to 9 columns displayed
pd.set_option("display.width", None) #auto-detect display width

cali_df = pd.DataFrame(cali.data, columns=cali.feature_names) #create dataframe with the data and the feature names
cali_df["MedHouseValue"] = pd.Series(cali.target) #add a column (a series) to a dataframe

print(cali_df.head()) #the head call is a peak at the first 5 rows

sample_df = cali_df.sample(frac=0.1, random_state=17) #get a fraction of the data
                                                      #Use Dataframe sample method to randomly select 10% of the 20,640 samples to graph it

import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set(font_scale=2)
sns.set_style("whitegrid")

for feature in cali.feature_names:
    plt.figure(figsize=(8, 4.5)) #8 by 4.5 figure
    sns.scatterplot(
        data=sample_df,
        x=feature,
        y="MedHouseValue",
        hue="MedHouseValue", 
        palette="cool",
        legend=False,
    )

#plt.show() #gives 8 graphs with the x as each feature, the y as the median income, and the 2000 samples as the data

from sklearn.model_selection import train_test_split

train_data, test_data, train_target , test_target = train_test_split(cali.data, cali.target, random_state=11) #x is the data, y is the target 

from sklearn.linear_model import LinearRegression

lr = LinearRegression() 

lr.fit(X=train_data, y=train_target) #for all the samples i give you for x and y (75% of all the samples), learn these 

predicted = lr.predict(test_data) #using the test data (x-values) that is 25% , predict the test_target (y-values) 25% because we have trained our machine so it should know what to match
expected = test_target #these are the targets that we expect 

print(f"predicted:{predicted[::5]} expected: {expected[::5]}")
#predicted:[1.25396876 2.36316557 1.88733812 ... 1.82952958 2.74716001 1.74620044] expected: [0.762 2.393 2.021 ... 4.025 2.387 0.906]
#the algorithms in sklearn are not the most evolved, so these targets are off because of the module alogrithms themself, and because we don't have enough features

df = pd.DataFrame()

df["expected"] = pd.Series(expected)
df["predicted"] = pd.Series(predicted)

import matplotlib.pyplot as plt2
figure = plt2.figure(figsize=(9,9))

axes = sns.scatterplot(
    data=df,
    x="expected",
    y="predicted", 
    hue="predicted", 
    palette="cool", 
    legend=False
)

#set the x and y axes limits to use the same scale along both axes

start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())

axes.set_xlim(start, end)
axes.set_ylim(start, end)

line = plt2.plot([start, end], [start, end], "k--")

plt2.show()




