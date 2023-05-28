import numpy as np
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data.info()
test_data.info()

train_data.drop(["Cabin"], axis=1, inplace=True)
test_data.drop(["Cabin"], axis=1, inplace=True)

#Survived
# sns.countplot(train_data.Survived)
# plt.show()

sns.countplot(train_data.Pclass)
plt.show()
