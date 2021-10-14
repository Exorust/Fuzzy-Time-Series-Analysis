# Run the FIDES part for thermo and mesophilic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import fides_level
import fides_trend
# import os
#
# os.system('./fides_level')

df = pd.read_csv('completedata.csv')
print("data loaded")

print("\n For Thermophilic:")
# Features
X = pd.DataFrame(df, columns=['Days','pHT','AlkalinityT','BiogasT','Level_T','Trend_T'])
# print(X)
y = pd.DataFrame(df, columns=['TS_M'])

pd.load(fides_level)
df_final.insert(fides.use(X['Level_T']))
print("Level Calculated")
pd.load(fides_level)
df_final.insert(fides.use(X['Trend_T']))
print("Trend Calculated")

print("\n For Mesophilic:")
X = pd.DataFrame(df, columns=['Days','pHM','AlkalinityM','BiogasM','Level_M','Trend_M'])
pd.load(fides_trend)
df_final.insert(fides.use(X['Level_M']))
print("Level Calculated")
pd.load(fides_level)
df_final.insert(fides.use(X['Trend_M']))
print("Trend Calculated")

df_final.csv("data_output.csv")
print("data saved")
