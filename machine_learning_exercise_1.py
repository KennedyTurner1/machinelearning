from sklearn.datasets import fetch_california_housing #20648 samples

cali = fetch_california_housing() #bunch object 
                                  #basically a pandas pataframe

import pandas as pd 

pd.set_option("precision", 4) 
pd.set_option("max_columns", 9) #up to 9 columns displayed
pd.set_option("display.width", None) #auto-detect display width

cali_df = pd.DataFrame(cali.data, columns=cali.feature_names) #create dataframe with the data and the feature names
cali_df["MedHouseValue"] = pd.Series(cali.target) #add a column (a series) to a dataframe

sample_df = cali_df.sample(frac=0.1, random_state=17) #get a fraction of the data
                                                      #Use Dataframe sample method to randomly select 10% of the 20,640 samples to graph it

import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set(font_scale=1.1)
sns.set_style("whitegrid")

grid = sns.pairplot(data=sample_df, vars=cali_df.columns[:8], hue='MedHouseValue')
plt.show()