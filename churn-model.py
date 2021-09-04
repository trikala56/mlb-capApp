import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
# relax display limits
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df_churn = pd.read_csv(r'telsco_churn.csv')
print(df_churn.head())
# filter columns
df_churn = df_churn[['gender', 'PaymentMethod', 'MonthlyCharges', 'tenure',
'Churn']].copy()

# new dataframe and replace missing values
df = df_churn.copy()
df.fillna(0, inplace=True)

# create dummy variablers for our categorical columns Gender and PaymentMethods
encode = ['gender', 'PaymentMethod']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

# map the Yes and No values to 1 & 0
df['Churn'] = np.where(df['Churn']=='Yes', 1, 0)
# define the feature and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# define the model
clf = RandomForestClassifier()
clf.fit(X, y)

# save model to pickle
pickle.dump(clf, open('churn_clf.pkl', 'wb'))
