import numpy as np
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# train_data.info()
# test_data.info()

train_data.drop(["Cabin"], axis=1, inplace=True)
test_data.drop(["Cabin"], axis=1, inplace=True)

# Survived Passengers (800)
# sns.countplot(train_data.Survived)
# plt.show()

# Passenger class
# sns.countplot(train_data.Pclass)
# plt.show()
# sns.barplot(x="Pclass", y="Survived", data=train_data)
# plt.show()

# plt.hist(train_data.Age, edgecolor='black')
# plt.xlabel('Age')
# plt.ylabel('count')
# plt.show()

# sns.boxplot(x='Survived', y='Age', data=train_data)
# plt.show()

# train_data.Ticket.head(10)

# train_data.Name.head(10)

# AgeMedian_by_titles = train_data.groupby('Title')['Age'].median()
# AgeMedian_by_titles

# for title in AgeMedian_by_titles.index:
#     train_data['Age'][(train_data.Age.isnull()) & (train_data.Title == title)] = AgeMedian_by_titles[title]
#     test_data['Age'][(test_data.Age.isnull()) & (test_data.Title == title)] = AgeMedian_by_titles[title]

# sns.distplot(train_data.Fare)
# plt.show()