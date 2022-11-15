#%%
from tkinter import Label
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)



# print("***** Train_Set *****")
# print(train.describe())
# print("\n")
# print("***** Test_Set *****")
# print(test.head())

train.isna().head()
test.isna().head()

#get total number of values missing in both datasets
#total_missing_train = train.isna().sum()
#total_missing_test = test.isna().sum()
#print("Total missing values in train set: ", total_missing_train)
#print("Total missing values in test set: ", total_missing_test)

#fill missing values with mean
train.fillna(train.mean(numeric_only=True), inplace=True)
test.fillna(test.mean(numeric_only=True), inplace=True)

#survival count by passenger class
#train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(train, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

#drop unnecessary columns
train.drop(['Cabin', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)
test.drop(['Cabin', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])

train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

train.info()
#test.info()

#drop survived column from train and save in variable x
x = np.array(train.drop(['Survived'], 1).astype(float))
y = np.array(train['Survived'])

train.info()

kmeans = kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
kmeans.fit(x)

correct = 0

for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(x))

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
kmeans.fit(x_scaled)

correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(x))