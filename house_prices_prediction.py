import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import mean_squared_log_error


train = pd.read_csv(r"C:\Users\z003zvfc\Downloads\train.csv")
test = pd.read_csv(r"C:\Users\z003zvfc\Downloads\test.csv")

ntrain = train.shape[0]

all_data = pd.concat((train, test), sort=False)

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

train = all_data[:ntrain]
test = all_data[ntrain:]

x_train = train.drop(['SalePrice', 'Id'], axis=1)
y = train.loc[:,'SalePrice']
x_test = test.drop(['SalePrice', 'Id'], axis=1)

lm = linear_model.LinearRegression()
model = lm.fit(x_train, y)
test_predictions = model.predict(x_test)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': test_predictions})
my_submission.to_csv('bahadir_final5.csv', index=False)
