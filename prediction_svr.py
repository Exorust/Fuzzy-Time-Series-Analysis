import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import metrics

df = pd.read_csv('completedata.csv')
print("data loaded")

# Features
X = pd.DataFrame(df, columns=['Days','pHT','AlkalinityT','BiogasT','Level_T','Trend_T'])
# print(X)
y = pd.DataFrame(df, columns=['TS_M'])
# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train)

#SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
print("done SVR Load")
y_rbf_test = svr_rbf.fit(X_train, y_train.values.ravel()).predict(X_test)
print("done fit")

results_rbf = metrics.r2_score(y_test.values.ravel(), y_rbf_test)
print('R^2 value:', metrics.r2_score(y_test.values.ravel(), y_rbf_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test.values.ravel(), y_rbf_test))
print('Mean Squared Error:', metrics.mean_squared_error(y_test.values.ravel(), y_rbf_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test.values.ravel(), y_rbf_test)))
