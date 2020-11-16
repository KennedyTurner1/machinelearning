import pandas as pd 

#x and y 
train = pd.read_csv("animals_train.csv")
df_train = pd.DataFrame(train.iloc[:, :16])
df_target = pd.Series(train.iloc[:, 16])

#testing the training module
test = pd.read_csv("animals_test.csv")
df_test = pd.DataFrame(test.iloc[:, 1:])

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

knn.fit(X=df_train, y=df_target)
predicted = knn.predict(df_test)

df_output = pd.DataFrame()
df_output["class_number"] = pd.Series(predicted)

my_dict = {1: "Mammal", 2: "Bird", 3: "Reptile", 4: "Fish", 5: "Amphibian", 6: "Bug", 7: "Invertibrate"}
types = []

for index, key in df_output["class_number"].items():
    value = my_dict.get(key, "no num")
    types.append(value)

df_output["class_type"] = types

print(df_output)

#df_output.to_csv(r"")
