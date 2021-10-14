import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPRegressor

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
regressor = MLPRegressor(activation='relu',hidden_layer_sizes=(6,6,6,2),max_iter=1000,random_state=0)
print("done Load")
y_pred_test = regressor.fit(X_train, y_train.values.ravel()).predict(X_test)
print("done fit")

print('R^2 value:', metrics.r2_score(y_test.values.ravel(), y_pred_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test.values.ravel(), y_pred_test))
print('Mean Squared Error:', metrics.mean_squared_error(y_test.values.ravel(), y_pred_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test.values.ravel(), y_pred_test)))
