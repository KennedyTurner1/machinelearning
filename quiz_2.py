import pandas as pd 

classes = pd.read_csv("animal_classes.csv", usecols=["Class_Number","Class_Type"])
train = pd.read_csv("animals_train.csv")

df = pd.DataFrame(classes)
df_2 = pd.DataFrame(train)

series = pd.Series(df_2["class_number"])
my_dict = {1: "Mammal", 2: "Bird", 3: "Reptile", 4: "Fish", 5: "Amphibian", 6: "Bug", 7: "Invertibrate"}

types = []

for index, key in df_2["class_number"].items():
    value = my_dict.get(key, "no num")
    types.append(value)

df_2["class_type"] = types

del df_2["class_number"]

print(df_2)
'''
from sklearn.linear_model import LinearRegression

lr = LinearRegression() 

lr.fit(X=df_2, y=df_2["class_type"])
'''