import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor


ad = pd.read_csv("daily_best_features.csv", header = 0)
tg = pd.read_csv("MSB.AX_daily.csv", header = 0)

Xr = ad[['SHL_Close','USIR%','ANN_Close', 'RMD_Close']]
yr = tg.Close.values

model = LinearRegression()

# reg_scores = cross_val_score(...
reg_scores = cross_val_score(model, Xr, yr, cv=4)

#get scores
print("Lin Reg_Scores: \n", reg_scores, np.mean(reg_scores))

# fit models
linreg = LinearRegression().fit(Xr, yr)

# fit a Pandom Forest Regressor to the data
X_train, X_test, y_train, y_test = train_test_split(Xr, yr, test_size=0.3)

#fit a base model
forest = RandomForestRegressor()
_ = forest.fit(X_train, y_train)

print(f"R2 for training set: {forest.score(X_train, y_train)}")
print(f"R2 for testing set: {forest.score(X_test, y_test)}\n")
