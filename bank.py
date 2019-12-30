from pandas import read_csv
import pandas
import numpy
from matplotlib import pyplot as plt
import seaborn

filename = 'bank-additional-full.csv'
data = read_csv(filename, delimiter=";")

array = data.values
print(len(array[0]))
# print(array[0:10, :])
X = array[:, 0:20]
Y = array[:, 20]
print(X[0, :], Y[0])

correlations = data.corr()
seaborn.heatmap(correlations)
plt.show()

