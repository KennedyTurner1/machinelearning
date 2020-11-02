from sklearn.datasets import load_digits #access to the sample data

digits = load_digits()

#print(digits.DESCR) #describe the dataset 
'''
print(digits.data[:2])  #value between 0 and 16 for each of the 64 features
print(digits.data.shape) #(1797, 64) 1797 rows (samples), 64 columns (features)
print(digits.target[:2]) #[4 0 5 3 6 9 6 1 7 5 4 4 7 2 8 2 2 5 7 9] this is what the 
print(digits.target.shape) #(1797,) there is only one target in which all the columns in the row describe one thing

print(digits.images[:2]) #this is an 8x8 representation of the features that helps you visualize the data
'''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(6,4)) 

#python zip function bundles 3 iterabales and produces 1 iterable
for item in zip(axes.ravel(), digits.images, digits.target):
    axes,image,target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()
#plt.show()

from sklearn.model_selection import train_test_split #when we give the data, the target, nd make it in a random state,
                                                     #it will return a tuple with 4 things
                                                     #x train, x test, y train, y test
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11) #75% training, 25% testing
print(x_train.shape) #(1347, 64) 75%, represents the data, these are numpy arrays
print(y_train.shape) #(1347,) 75%, represents the target
print(x_test.shape) #(450, 64) 25%, represents the data
print(y_test.shape) #(450,) 25%, represents the target

from sklearn.neighbors import KNeighborsClassifier #classification model we want to use to train/test
                                                   #classify every sample from 0-9
knn = KNeighborsClassifier()

#load the training data into the model using the fit method
knn.fit(X=x_train, y=y_train) #give it the data as the x and the target at the y 
                             #this is the machine learning, the fit method
            
#feed it test data to see if it can correctly predict the outcome
predicted = knn.predict(x_test) #just give it the x value because we want it to predict the y value. 

expected = y_test #let's see how they compare, what is the target value we were supposed to test  

print(predicted[:20]) #[0 4 9 9 3 1 4 1 5 0 4 9 4 1 5 3 3 8 5 6]
print(expected[:20])  #[0 4 9 9 3 1 4 1 5 0 4 9 4 1 5 3 3 8 3 6]

wrong = [(p,e) for (p,e) in zip(predicted[:20], expected[:20]) if p != e] #iterating through both of the arrays and seeing if they are the same

print(wrong) #[(5, 3)]

print(format(knn.score(x_test, y_test), ".2%")) #looking for how accurate the x and the y test are, 97.78% accurate

from sklearn.metrics import confusion_matrix

cf = confusion_matrix(y_true=expected, y_pred=predicted) #what is the target you were supposed to get, what you actually got

print(cf) #the number 8 out of 0-9 has the most issues so we think this is where it predicted it wrong

import pandas as pd
import seaborn as shs
import matplotlib.pyplot as plt2

cf_df = pd.DataFrame(cf, index=range(10)) #rows (digits 0-9) are 9 rows, columns
fig = plt2.figure(figsize=(7,6))
axes = shs.heatmap(cf_df, annot=True, cmap= plt2.cm.nipy_spectral_r) 
plt2.show()