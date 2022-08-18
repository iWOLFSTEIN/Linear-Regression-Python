import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os


dataframe = pd.read_csv('E:\Visual Code Projects\Python Projects\python\Machine Learning\datasets\\real_estate.csv')
print('\n\nFirst 5 Rows of our Dataframe\n')
print(dataframe.head())

predictor = dataframe.drop(['Y house price of unit area'], axis='columns')
target = dataframe['Y house price of unit area']

X_train, X_test, y_train, y_test = train_test_split(predictor, target, train_size =0.8)
model = LinearRegression()
model.fit(X_train, y_train)

predict = model.predict(X_test)
print(predict)

print(model.score(X_test, y_test))
os.system("pause")
