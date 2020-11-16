import pandas as pd 

#x and y 
train = pd.read_csv("animals_train.csv")
df_train = pd.DataFrame(train.iloc[:, :16])
df_target = pd.Series(train.iloc[:, 16])

#testing the training module
test = pd.read_csv("animals_test.csv")
df_test = pd.DataFrame(test.iloc[:, 1:])

#animal_names
df_animaldata = pd.DataFrame(test.iloc[:, 1:])
df_animaltarget = pd.Series(test.iloc[:, 0])

from sklearn.neighbors import KNeighborsClassifier
knn_an = KNeighborsClassifier()
knn_an.fit(X=df_animaldata, y=df_animaltarget)
predicted_an = knn_an.predict(df_test)

df_output = pd.DataFrame()
df_output["animal_name"] = pd.Series(predicted_an)

knn = KNeighborsClassifier()
knn.fit(X=df_train, y=df_target)
predicted = knn.predict(df_test)

df_output["class_number"] = pd.Series(predicted)

my_dict = {1: "Mammal", 2: "Bird", 3: "Reptile", 4: "Fish", 5: "Amphibian", 6: "Bug", 7: "Invertebrate"}
types = []

for index, key in df_output["class_number"].items():
    value = my_dict.get(key, "no num")
    types.append(value)

df_output["prediction"] = types
del df_output["class_number"]

print(df_output)

df_output.to_csv(r"C:\Users\Kennedy's PC\Documents\Baylor University\Fall 2020 - Senior Year\MIS 4322\machinelearning\outcomes.csv", index=False, header=True)
