import pandas as pd 

classes = pd.read_csv("animal_classes.csv", usecols=["Class_Number","Class_Type"])
train = pd.read_csv("animals_train.csv")
test = pd.read_csv("animals_test.csv")


df_class = pd.DataFrame(classes)
df_train = pd.DataFrame(train)
df_test = pd.DataFrame(test)

my_dict = {1: "Mammal", 2: "Bird", 3: "Reptile", 4: "Fish", 5: "Amphibian", 6: "Bug", 7: "Invertibrate"}

types = []

for index, key in df_train["class_number"].items():
    value = my_dict.get(key, "no num")
    types.append(value)

df_train["class_type"] = types
del df_train["class_number"]

print(df_train)

from sklearn.linear_model import LinearRegression

lr = LinearRegression() 

lr.fit(X=df_train, y=df_train["class_type"])

test_data = df_test.loc[0:36, "hair":"catsize"]
test_target = pd.Series(df_test["animal_name"])

predicted = lr.predict(test_data)
expected = test_target

df_output = pd.DataFrame()

df_output["expected"] = pd.Series(expected)
df_output["predicted"] = pd.Series(predicted)

print(df_output)

df_output.to_csv(r"C:\Users\Kennedy'sPC\Documents\Baylor University\Fall 2020 - Senior Year\MIS 4322\machinelearning\outcomes.csv", index=False, header=True)