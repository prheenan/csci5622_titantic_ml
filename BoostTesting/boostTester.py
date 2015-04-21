from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
import csv as csv

fare_ceiling = 40

# Importing the train file and removing the null fields

df = pd.read_csv('../data/train.csv',header=0)
df['Gender']=df['Sex'].map({'female':0, 'male': 1}).astype(int)
median_ages = np.zeros((2,3))
for i in range (0,2):
    for j in range (0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()
df['AgeFill'] = df['Age']
for i in range (0,2):
    for j in range (0,3):
        df.loc[(df.Age.isnull())&(df.Gender == i)&(df.Pclass == j+1),'AgeFill'] = median_ages[i,j]
df.loc[df.Fare >= fare_ceiling, 'Fare'] = fare_ceiling-1
df['bFare'] = df['Fare']
df.loc[(df['bFare'] < 10),'bFare'] = 3
df.loc[(df['bFare'] > 10)&(df['bFare'] < 20),'bFare'] = 2
df.loc[(df['bFare'] > 20)&(df['bFare'] < 30),'bFare'] = 1
df.loc[(df['bFare'] > 30),'bFare'] = 0
#df = df.dropna()
train_data = df.values

# Importing the test file

df2 = pd.read_csv('../data/test.csv',header=0)
df2['Gender']=df2['Sex'].map({'female':0, 'male': 1}).astype(int)
#df2 = df2.dropna()
df2['Fare'].fillna(df['Fare'].mean(), inplace=True) # replaces the null value with 0, probably should use a better method
df2['AgeFill']=df2['Age']
for i in range (0,2):
    for j in range (0,3):
        df2.loc[(df2.Age.isnull())&(df2.Gender == i)&(df2.Pclass == j+1),'AgeFill'] = median_ages[i,j]
df2.loc[df2.Fare >= fare_ceiling, 'Fare'] = fare_ceiling-1
df2['bFare'] = df2['Fare']
df2.loc[(df2['bFare'] < 10),'bFare'] = 3
df2.loc[(df2['bFare'] > 10)&(df2['bFare'] < 20),'bFare'] = 2
df2.loc[(df2['bFare'] > 20)&(df2['bFare'] < 30),'bFare'] = 1
df2.loc[(df2['bFare'] > 30),'bFare'] = 0
test_data = df2.values

# Fitting the classifier and running it

clf=AdaBoostClassifier()
clf.fit(train_data[:,[12,2]],train_data[:,1])
results = clf.predict(test_data[:,[11,1]])

# Writing out the results

prediction_file = open("genderbasedmodel2.csv", "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])
counter = 0
for row in test_data:
    prediction_file_object.writerow([row[0],results[counter]])
    counter = counter + 1
prediction_file.close()
