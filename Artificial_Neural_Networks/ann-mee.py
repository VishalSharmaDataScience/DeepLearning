# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:00:37 2019

@author: inspiron
"""

import os
os.getcwd()
os.chdir("H:\Deep L\P16-Artificial-Neural-Networks\Artificial_Neural_Networks")

#check if tensorflow using CPU or GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#part1 data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values



# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#dummy variable for country olumn as 1 is not less than 2 here
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#country is now into 3 variables dummy so we have to get away from dummy variable tap we remove 1 dummy column
X = X [:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#PART 2
####Now lets make ann
#import keras and package 
import keras
from keras.models import Sequential
from keras.layers import Dense



#Initialize the ann , can be done by 2 ways . we will do the by defining the sequential layers
classifier = Sequential()



# Create your classifier here
#now layer addition , we have 6 pointsto consider
#kernel initializer close to 0 initial weights if uniform
#no. of inputnodes = 11 , for hidden is average so 11 + 1 by 2 = 6 
# , activation function for neuron = rectifier(0,1) or sigmoid (probab)
#sigmoidfor out putlayer and rectifier for hidden layers
#compare y , back propogate , learning rate then weight update, reinforced or batch = 1epoch 

classifier.add(Dense(units = 6, input_dim = 11 ,kernel_initializer = "uniform",activation = "relu"))
#1 layer added now new layer 
#again avg will be the hidden layer nodes
classifier.add(Dense(units = 6 ,kernel_initializer = "uniform",activation = "relu"))

#1 lasthidden layer

classifier.add(Dense(units = 1 ,kernel_initializer = "uniform",activation = "sigmoid"))
#if more than 1 categories then unit(one hot coded) and activation(softmax) will change 



#Compiling the ANN = appling stochastic gradient  or batch
#optimier = to getthe best weights . adam is a type of stochastic gradient 
#loss = loss function within adam function  and logarithmic loss function for binary or multiple (_categoricalcross_entropy)
classifier.compile(optimizer= "adam" , loss = "binary_crossentropy",metrics = ["accuracy"])




#PART3
# Fitting classifier to the Training set

classifier.fit(X_train,y_train,batch_size= 10 ,epochs = 100 )

# Predicting the Test set results
y_pred = classifier.predict(X_test) 
y_pred = (y_pred>0.5)
#forconfusion matrix convert to 0 or 1 .



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#accuracy = correct /total = 1689/2000 = 0.8445

#nw prediction sametransformation as of data
new_prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
#complete


#PART 4
#Now if we re run results are different so more performance evaluation and all required
# so k fold classication  

#Part4.1 preprocessing 
#run part 1 of this
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, input_dim = 11 ,kernel_initializer = "uniform",activation = "relu"))
    classifier.add(Dense(units = 6 ,kernel_initializer = "uniform",activation = "relu"))
    classifier.add(Dense(units = 1 ,kernel_initializer = "uniform",activation = "sigmoid"))
    classifier.compile(optimizer= "adam" , loss = "binary_crossentropy",metrics = ["accuracy"])
    return classifier


classifier = KerasClassifier(build_fn = build_classifier,batch_size= 10 ,epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train,y = y_train, cv=10,
                    n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()


#complete k fold Cv we got avg of 83% accuracy so now we 
#can say it gives around this much . Now we need to regularize it to reduce overfitting

# Now we might have  overfitted high training accu lowtest accu and also ifvarianceis high 
#DROpOUT REgularization = some neurons are not fired randomly each time 
from keras.layers import DropOut
#4.2 creating ANNwith dropout  in whatever number of hidden layers you want you can add 
classifier = Sequential()



classifier.add(Dense(units = 6, input_dim = 11 ,kernel_initializer = "uniform",activation = "relu"))
classifier.add(DropOut(p=0.1))
classifier.add(Dense(units = 6 ,kernel_initializer = "uniform",activation = "relu"))
classifier.add(DropOut(p=0.1))
#restcode same 
#if more than 1 categories then unit(one hot coded) and activation(softmax) will change 



#Compiling the ANN = appling stochastic gradient  or batch
#optimier = to getthe best weights . adam is a type of stochastic gradient 
#loss = loss function within adam function  and logarithmic loss function for binary or multiple (_categoricalcross_entropy)
classifier.compile(optimizer= "adam" , loss = "binary_crossentropy",metrics = ["accuracy"])




#PART3
# Fitting classifier to the Training set

classifier.fit(X_train,y_train,batch_size= 10 ,epochs = 100 )

# Predicting the Test set results
y_pred = classifier.predict(X_test) 
y_pred = (y_pred>0.5)
#forconfusion matrix convert to 0 or 1 .



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#accuracy = correct /total = 1689/2000 = 0.8445

#nw prediction sametransformation as of data
new_prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
#complete







#PARAMETER TUNING  using gridsearch on hyperparameters
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, input_dim = 11 ,kernel_initializer = "uniform",activation = "relu"))
    classifier.add(DropOut(p=0.1))
    classifier.add(Dense(units = 6 ,kernel_initializer = "uniform",activation = "relu"))
    classifier.add(DropOut(p=0.1))
    classifier.add(Dense(units = 1 ,kernel_initializer = "uniform",activation = "sigmoid"))
    classifier.compile(optimizer= optimizer , loss = "binary_crossentropy",metrics = ["accuracy"])
    return classifier

#creategridfor hyperpara epochs and batch size 
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {
            "batch_size":[25,32],
             "epochs":[100,500],
             "optimizer":["adam","rmsprop"]
             }

grid_search = GridSearchCV(estimator = classifier , param_grid = parameters, scoring = "accuracy",cv = 10)

grid_search.fit(X_train,y_train)

#bestparam and best accu 
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
