
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#read data file
df = pd.read_csv('Linear_Reg_Sales.csv')

advert = df[['Advert']]
sales = df['Sales']

#default split ration is 25% for test set, hardcode ran_state: everytime it runs, the same random value is generated
x_train, x_test,y_train, y_test = train_test_split(advert, sales, random_state=1)

linReg=LinearRegression()

#fit linear model to the train data set
# under the current version, linReg.fit(x_train,y_train) will produce a warning
linReg.fit(x_train.values,y_train)

# deploy to the flask server
# flask server need to be started
pickle.dump(linReg, open('model1.pkl', 'wb'))  #serialize the object

