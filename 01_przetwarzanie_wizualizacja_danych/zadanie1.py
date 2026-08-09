import pandas as pd
import numpy as np
import math as m
titanic_data=pd.read_csv("titanic.csv", index_col=0)
print(titanic_data.iloc[3:6, [2,5]])
# print(titanic_data.dtypes)
# print(titanic_data.describe(include='all'))
# print(titanic_data['SibSp'].unique())
# print(titanic_data['Survived'].value_counts())
# sex_in_class=titanic_data[['Pclass','Sex']].value_counts()
# print(sex_in_class[[(3,'male'), (1,'female')]])
# print(titanic_data.groupby('Sex')[['Survived']].mean())
# print(titanic_data.groupby('Pclass')[['Survived']].mean())
#print(titanic_data.groupby('Pclass')[['Fare', 'Age']].max())

# print(titanic_data.head())
# print(titanic_data.info())