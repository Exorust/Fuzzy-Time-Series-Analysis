import pandas
import numpy as np
import matplotlib.pyplot as plt

df = pandas.read_csv('newdata.csv')
# print(df)
X = df["Days"].values
# print(X)
Y_Mactual = df["Mactual"].values
Y_Mpred = df["Mpred"].values
Y_Tactual = df["Tactual"].values
Y_Tpred = df["Tpred"].values

plt.plot(X,Y_Mactual,color="red",label="Actual Values")
plt.plot(X,Y_Mpred, color="blue",label="Predicted Values")
plt.suptitle('Mesophilic TS in Waste Solids', fontsize=20)
plt.xlabel('Days', fontsize=14)
plt.ylabel('TS (g/l)', fontsize=14)
plt.legend(loc='upper left')

# plt.show()
plt.savefig('Meso_fides.jpg')
