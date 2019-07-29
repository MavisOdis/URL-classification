"""
Created on Sat Jul 27 20:31:48 2019

@author: Mavis
"""
"""import the necessary libraries"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

"""read the data into python"""
data = pd.read_csv("tadata.csv")

"""Split the data into feature matrice and response"""

X = data["url"]
y = data.label

"""split the data into training and testing data"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

Reg = LogisticRegression()
Reg.fit(X_train_dtm,y_train)


d =['gogle.']
d_dtm = vect.transform(d)
print(Reg.predict(d_dtm)[0])

z = ['uniport.edu.ng']
z_dtm =vect.transform(z)
print(Reg.predict(z_dtm)[0])
