import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor


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

regr = DecisionTreeRegressor(max_depth=9)
model = regr.fit(x_train, y)
test_predictions = model.predict(x_test)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': test_predictions})
my_submission.to_csv('bahadir_final1.csv', index=False)